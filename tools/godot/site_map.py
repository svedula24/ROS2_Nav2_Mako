#!/usr/bin/env python3
"""Nav2 static map for a Godot sim site: the collision mesh fused with the site's occupancy.bin.

  python3 site_map.py --mesh collision.ply --site <sites>/circuit4_cam_mesh --out tools/sim/godot_circuit4_map
  -> <out>.pgm + <out>.yaml (nav2 map_server, frame = the site's ENU = Nav2 `map`) + <out>_check.png

Two sources, each covering the other's weakness (both in the site's ENU frame, enu_shift 0):
  collision mesh   what the robot physically hits (the sim collides against it): precise geometry, but
                   holes where the capture missed a surface.
  occupancy.bin    1 m voxels of splat density (the sim's own planner grid): coarse and floater-prone,
                   but it knows where floor and walls were captured at all.
Fusion, at --res (0.05 m):
  1. obstacle = mesh surface inside the robot's band (floor + 0.15 .. floor + 1.0 m).
  2. known floor = mesh floor surface (floor +- 0.10) OR an occupancy floor voxel; known & not obstacle = free,
     anything else = unknown.
  3. gap fill: an occupancy voxel at robot height (0-1 m) whose 1 m column has NO mesh obstacle, where the
     mesh saw (almost) no floor (a capture hole, not an open aisle), and whose neighbouring columns have mesh
     obstacles (it continues a wall): its cells the mesh didn't see become obstacle. Isolated voxels are
     floaters and are dropped.
  4. obstacle specks smaller than --min-speck cells are removed.
"""
import argparse
import json
import math
import os
import struct

import numpy as np

UNKNOWN, FREE, OCC = 205, 254, 0                 # map_server trinary pgm values (negate 0)
BIAS, MASK = 1 << 20, (1 << 21) - 1


# ------------------------------------------------------------------ readers
def read_ply(path):
    """Binary little-endian PLY with float x,y,z vertices and uchar-count int faces -> (V (n,3), F (m,3))."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        body = f.read()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("only binary_little_endian PLY is supported")
    nv = int(next(l.split()[2] for l in header if l.startswith("element vertex")))
    nf = int(next(l.split()[2] for l in header if l.startswith("element face")))
    # vertex block: count the scalar (float) properties declared right after `element vertex`
    i = next(k for k, l in enumerate(header) if l.startswith("element vertex")) + 1
    nprop = 0
    while header[i].startswith("property") and "list" not in header[i]:
        nprop += 1
        i += 1
    V = np.frombuffer(body, "<f4", count=nv * nprop).reshape(nv, nprop)[:, :3].astype(np.float64)
    off = nv * nprop * 4
    face = np.frombuffer(body, np.dtype([("n", "u1"), ("i", "<i4", 3)]), count=nf, offset=off)
    if not np.all(face["n"] == 3):
        raise ValueError("only triangle faces are supported")
    return V, face["i"].astype(np.int64)


def _variant(buf, p):
    """Minimal Godot 4 Variant decoder (the types occupancy.bin uses). -> (value, new offset)."""
    head = struct.unpack_from("<I", buf, p)[0]
    t, wide = head & 0xFFFF, bool(head & (1 << 16))
    p += 4
    if t == 1:                                   # bool
        return bool(struct.unpack_from("<I", buf, p)[0]), p + 4
    if t == 2:                                   # int
        return (struct.unpack_from("<q", buf, p)[0], p + 8) if wide else (struct.unpack_from("<i", buf, p)[0], p + 4)
    if t == 3:                                   # float
        return (struct.unpack_from("<d", buf, p)[0], p + 8) if wide else (struct.unpack_from("<f", buf, p)[0], p + 4)
    if t == 4:                                   # String: u32 length + utf8, padded to 4
        n = struct.unpack_from("<I", buf, p)[0]
        s = buf[p + 4:p + 4 + n].decode("utf-8")
        return s, p + 4 + n + (-n % 4)
    if t == 27:                                  # Dictionary
        n = struct.unpack_from("<I", buf, p)[0] & 0x7FFFFFFF
        p += 4
        d = {}
        for _ in range(n):
            k, p = _variant(buf, p)
            v, p = _variant(buf, p)
            d[k] = v
        return d, p
    if t == 31:                                  # PackedInt64Array
        n = struct.unpack_from("<I", buf, p)[0]
        return np.frombuffer(buf, "<i8", count=n, offset=p + 4).copy(), p + 4 + 8 * n
    raise ValueError(f"unsupported Variant type {t} at byte {p - 4}")


def read_occupancy(path):
    """Godot store_var() Dictionary {voxel_m, min_count, plan_m, occupied, blocked_plan}.
    -> (voxel_m, cells (n,3) int Godot (gx, gy-up, gz))."""
    buf = open(path, "rb").read()
    n = struct.unpack_from("<I", buf, 0)[0]
    d, _ = _variant(buf[4:4 + n], 0)
    return float(d["voxel_m"]), decode_keys(d["occupied"])


def decode_keys(keys):
    k = np.asarray(keys, np.int64)
    return np.stack([((k >> 42) & MASK) - BIAS, ((k >> 21) & MASK) - BIAS, (k & MASK) - BIAS], 1)


def encode_keys(cells):
    c = np.asarray(cells, np.int64) + BIAS
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


# ------------------------------------------------------------------ geometry
def sample_triangles(V, F, step):
    """Points on every triangle, spaced <= step (barycentric grid per triangle). -> (n,3)."""
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    edge = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)])
    nseg = np.clip(np.ceil(edge / step).astype(int), 1, 64)
    out = []
    for n in np.unique(nseg):
        sel = nseg == n
        ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
        u, v = ij[:, 0][None, :, None], ij[:, 1][None, :, None]
        a, b, c = A[sel][:, None, :], B[sel][:, None, :], C[sel][:, None, :]
        out.append((a + u * (b - a) + v * (c - a)).reshape(-1, 3))
    return np.concatenate(out) if out else np.zeros((0, 3))


class Grid:
    def __init__(self, x0, y0, x1, y1, res):
        self.x0, self.y0, self.res = x0, y0, res
        self.w, self.h = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))

    def idx(self, x, y):
        """-> (col, row-from-south, inside mask)."""
        c = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        r = np.floor((np.asarray(y) - self.y0) / self.res).astype(int)
        return c, r, (c >= 0) & (r >= 0) & (c < self.w) & (r < self.h)

    def mark(self, arr, x, y):
        c, r, ok = self.idx(x, y)
        arr[r[ok], c[ok]] = True


def neighbours_any(mask_cols):
    """8-neighbourhood OR (excluding the cell itself) on a boolean 2D array."""
    out = np.zeros_like(mask_cols)
    p = np.pad(mask_cols, 1)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                out |= p[1 + dr:1 + dr + mask_cols.shape[0], 1 + dc:1 + dc + mask_cols.shape[1]]
    return out


def remove_specks(occ, min_cells):
    if min_cells <= 1:
        return occ
    try:
        from scipy import ndimage
    except ImportError:
        return occ
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    if n == 0:
        return occ
    sizes = ndimage.sum(occ, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_cells
    return keep[lab]


# ------------------------------------------------------------------ fusion
def build(V, F, occ_cells, voxel_m, bounds_xy, floor_z, res=0.05, band=(0.15, 1.0), floor_tol=0.10,
          wall_layers=(0,), gap_fill=True, max_floor_seen=0.2, min_speck=3):
    """-> dict(grid, map (h,w) uint8 north-up, stats, layers for the check image)."""
    (bx0, by0), (bx1, by1) = bounds_xy
    g = Grid(bx0, by0, bx1, by1, res)
    pts = sample_triangles(V, F, res * 0.7)
    z = pts[:, 2]
    obstacle = np.zeros((g.h, g.w), bool)
    floor = np.zeros((g.h, g.w), bool)
    g.mark(obstacle, *pts[(z >= floor_z + band[0]) & (z <= floor_z + band[1])][:, :2].T)
    g.mark(floor, *pts[np.abs(z - floor_z) <= floor_tol][:, :2].T)
    mesh_obstacle = obstacle.copy()

    # occupancy voxels -> ENU 1 m columns. Godot cell (gx, gy, gz) covers ENU x [gx, gx+1),
    # north (-gz-1, -gz], up [gy, gy+1) (times voxel_m).
    occ_floor = np.zeros((g.h, g.w), bool)
    gap = np.zeros((g.h, g.w), bool)
    v = voxel_m
    cells = np.asarray(occ_cells, int).reshape(-1, 3)
    floor_gy = math.floor(floor_z / v)
    k = max(1, int(round(v / res)))
    sub = (np.arange(k) + 0.5) * res                  # fine-cell centres inside one voxel column

    def column_cells(gx, gz):
        xs = gx * v + sub
        ys = -(gz + 1) * v + sub
        X, Y = np.meshgrid(xs, ys)
        return X.ravel(), Y.ravel()

    for gx, gz in {(c[0], c[2]) for c in cells if c[1] == floor_gy}:
        g.mark(occ_floor, *column_cells(gx, gz))
    known = floor | occ_floor

    gap_cols = 0
    if gap_fill and len(cells):
        wall = {(c[0], c[2]) for c in cells if c[1] in wall_layers}      # Godot y 0..2 m: wall height
        # coarse 1 m map of mesh obstacles, for "is this column empty / do neighbours have walls"
        gx0, gz0 = int(math.floor(bx0 / v)) - 1, int(math.floor(-by1 / v)) - 1
        W, H = int(math.ceil((bx1 - bx0) / v)) + 3, int(math.ceil((by1 - by0) / v)) + 3
        coarse = np.zeros((H, W), bool)
        oy, ox = np.nonzero(mesh_obstacle)
        cx = np.floor((g.x0 + (ox + 0.5) * res) / v).astype(int) - gx0
        cz = np.floor(-(g.y0 + (oy + 0.5) * res) / v).astype(int) - gz0
        okc = (cx >= 0) & (cz >= 0) & (cx < W) & (cz < H)
        coarse[cz[okc], cx[okc]] = True
        near_wall = neighbours_any(coarse)
        for gx, gz in wall:
            i, j = gz - gz0, gx - gx0
            if not (0 <= i < H and 0 <= j < W) or coarse[i, j] or not near_wall[i, j]:
                continue
            # A column where the mesh SAW floor is an open aisle, not a hole in a wall: splat density
            # at wall height there is furniture tops, people or ceiling fixtures (first run: the
            # unconditional fill walled off every corridor). Fill only where the mesh saw (almost) no
            # floor, and only the cells it didn't see.
            X, Y = column_cells(gx, gz)
            cc, rr, ok = g.idx(X, Y)
            cc, rr = cc[ok], rr[ok]
            if not len(cc) or floor[rr, cc].mean() > max_floor_seen:
                continue
            hole = ~floor[rr, cc]
            gap[rr[hole], cc[hole]] = True
            gap_cols += 1
    obstacle = remove_specks(obstacle | gap, min_speck)

    m = np.full((g.h, g.w), UNKNOWN, np.uint8)
    m[known] = FREE
    m[obstacle] = OCC
    stats = {"size": [g.w, g.h], "res": res, "origin": [g.x0, g.y0], "triangles": int(len(F)),
             "obstacle_cells": int(obstacle.sum()), "mesh_obstacle_cells": int(mesh_obstacle.sum()),
             "gap_fill_columns": gap_cols, "gap_cells": int((gap & ~mesh_obstacle).sum()),
             "free_cells": int((m == FREE).sum()), "unknown_cells": int((m == UNKNOWN).sum()),
             "floor_from_mesh": int(floor.sum()), "floor_added_by_occupancy": int((occ_floor & ~floor).sum())}
    return {"grid": g, "map": m[::-1].copy(), "stats": stats,
            "layers": {"mesh_obstacle": mesh_obstacle[::-1], "gap": gap[::-1], "mesh_floor": floor[::-1],
                       "occ_floor": occ_floor[::-1]}}


def write_map(result, out_base):
    g, m = result["grid"], result["map"]
    with open(out_base + ".pgm", "wb") as f:
        f.write(f"P5\n{g.w} {g.h}\n255\n".encode())
        f.write(m.tobytes())
    with open(out_base + ".yaml", "w", newline="\n") as f:
        f.write(f"# GENERATED by tools/godot/site_map.py: collision mesh + occupancy.bin fused (see its header)\n"
                f"image: {os.path.basename(out_base)}.pgm\nmode: trinary\nresolution: {g.res}\n"
                f"origin: [{g.x0:.4f}, {g.y0:.4f}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")


def write_check(result, path):
    """Side by side: mesh only | occupancy floor + mesh | fused (gap-filled cells in orange)."""
    L, m = result["layers"], result["map"]
    h, w = m.shape
    a = np.full((h, w, 3), 40, np.uint8); a[L["mesh_floor"]] = (200, 200, 200); a[L["mesh_obstacle"]] = (20, 20, 20)
    b = np.full((h, w, 3), 40, np.uint8); b[L["occ_floor"]] = (150, 170, 200); b[L["mesh_floor"]] = (200, 200, 200)
    b[L["mesh_obstacle"]] = (20, 20, 20)
    c = np.stack([m] * 3, -1).copy(); c[m == UNKNOWN] = (40, 40, 40); c[m == FREE] = (225, 225, 225)
    c[m == OCC] = (10, 10, 10); c[L["gap"] & (m == OCC)] = (240, 140, 20)
    sep = np.full((h, 6, 3), 255, np.uint8)
    img = np.concatenate([a, sep, b, sep, c], 1)
    try:
        import cv2
        cv2.imwrite(path, img[..., ::-1])
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(path)


def site_bounds(site_dir, pad=0.5):
    """ENU xy bounds from the site's blocks.json (bounds_godot) and its ground height."""
    b = json.load(open(os.path.join(site_dir, "blocks.json")))
    lo, hi = b["bounds_godot"]["min"], b["bounds_godot"]["max"]
    xs, ys = (lo[0], hi[0]), (-hi[2], -lo[2])           # ENU north = -Godot z
    ground = float(b.get("report", {}).get("ground_m", lo[1]))
    return ((min(xs) - pad, min(ys) - pad), (max(xs) + pad, max(ys) + pad)), ground


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True, help="the site's collision mesh (PLY, site ENU frame)")
    ap.add_argument("--site", required=True, help="the baked site folder (blocks.json, occupancy.bin)")
    ap.add_argument("--out", required=True, help="output base path (writes .pgm/.yaml/_check.png)")
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--band", type=float, nargs=2, default=(0.15, 1.0), help="obstacle band above the floor [m]")
    ap.add_argument("--no-gap-fill", action="store_true")
    ap.add_argument("--min-speck", type=int, default=3)
    a = ap.parse_args()
    bounds, ground = site_bounds(a.site)
    V, F = read_ply(a.mesh)
    occ = os.path.join(a.site, "occupancy.bin")
    voxel, cells = read_occupancy(occ) if os.path.exists(occ) else (1.0, np.zeros((0, 3), int))
    r = build(V, F, cells, voxel, bounds, ground, a.res, tuple(a.band), gap_fill=not a.no_gap_fill,
              min_speck=a.min_speck)
    write_map(r, a.out)
    write_check(r, a.out + "_check.png")
    print(json.dumps(r["stats"], indent=1))


if __name__ == "__main__":
    main()

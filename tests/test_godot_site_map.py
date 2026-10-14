"""tools/godot/site_map.py: collision mesh + occupancy.bin -> Nav2 map (synthetic inputs)."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "godot"))
import site_map as sm   # noqa: E402

FLOOR = -0.05          # like the real site (-0.07): inside the floor voxel layer gy = -1


def quad(x0, y0, x1, y1, z0, z1, vertical=True):
    """Two triangles: a vertical wall from (x0,y0) to (x1,y1) between z0..z1, or a horizontal patch."""
    if vertical:
        V = [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)]
    else:
        V = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    return np.array(V, float), np.array([[0, 1, 2], [0, 2, 3]])


def mesh(*parts):
    V, F, n = [], [], 0
    for v, f in parts:
        V.append(v); F.append(f + n); n += len(v)
    return np.concatenate(V), np.concatenate(F)


def store_var_dict(d):
    """Encode the occupancy dict the way Godot's store_var does (types used by occupancy.bin)."""
    def s(x):
        b = x.encode()
        return struct.pack("<II", 4, len(b)) + b + b"\0" * (-len(b) % 4)

    def v(x):
        if isinstance(x, float):
            return struct.pack("<If", 3, x)
        if isinstance(x, int):
            return struct.pack("<Ii", 2, x)
        if isinstance(x, np.ndarray):
            return struct.pack("<II", 31, len(x)) + x.astype("<i8").tobytes()
        raise TypeError(x)
    body = struct.pack("<II", 27, len(d)) + b"".join(s(k) + v(val) for k, val in d.items())
    return struct.pack("<I", len(body)) + body


def test_occupancy_bin_decoder_roundtrip(tmp_path):
    cells = np.array([[-3, -1, 2], [0, 0, 0], [5, 2, -7]])
    p = tmp_path / "occupancy.bin"
    p.write_bytes(store_var_dict({"voxel_m": 1.0, "min_count": 25, "plan_m": 2.0,
                                  "occupied": sm.encode_keys(cells), "blocked_plan": sm.encode_keys(cells[:1])}))
    voxel, back = sm.read_occupancy(str(p))
    assert voxel == 1.0 and back.tolist() == cells.tolist()


def test_ply_reader(tmp_path):
    V, F = quad(0, 0, 1, 0, 0, 1)
    p = tmp_path / "m.ply"
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    body = V.astype("<f4").tobytes() + b"".join(struct.pack("<B3i", 3, *f) for f in F)
    p.write_bytes(head.encode() + body)
    V2, F2 = sm.read_ply(str(p))
    assert np.allclose(V2, V) and F2.tolist() == F.tolist()


def build(occ_cells=(), **kw):
    # floor 4 x 3 m (the near side of the wall only) with a 1 x 1 m hole at x 1..2, y 1..2; a wall along
    # y = 3 with a gap between x 1.9 and 3.1 (the capture missed it; nothing seen beyond the wall)
    floor_parts = [quad(0, 0, 4, 1, FLOOR, FLOOR, False), quad(0, 2, 4, 3, FLOOR, FLOOR, False),
                   quad(0, 1, 1, 2, FLOOR, FLOOR, False), quad(2, 1, 4, 2, FLOOR, FLOOR, False)]
    wall = [quad(0, 3.02, 1.9, 3.02, FLOOR, 2.0), quad(3.1, 3.02, 4, 3.02, FLOOR, 2.0)]
    V, F = mesh(*floor_parts, *wall)
    return sm.build(V, F, np.array(occ_cells, int).reshape(-1, 3), 1.0, ((-0.5, -0.5), (4.5, 4.5)), FLOOR,
                    res=0.1, min_speck=1, **kw)


def cell(r, x, y):
    g = r["grid"]
    c, row, _ = g.idx(x, y)
    return r["map"][g.h - 1 - int(row), int(c)]                 # map is north-up


def test_floor_free_wall_occupied_outside_unknown():
    r = build()
    assert cell(r, 0.5, 0.5) == sm.FREE
    assert cell(r, 0.5, 3.0) == sm.OCC
    assert cell(r, 1.5, 1.5) == sm.UNKNOWN                     # mesh floor hole, no occupancy help
    assert cell(r, -0.3, -0.3) == sm.UNKNOWN


def test_occupancy_floor_voxel_fills_the_mesh_floor_hole():
    # Godot cell (gx=1, gy=-1, gz=-2) covers ENU x 1..2, north 1..2, up -1..0 (floor layer)
    r = build([(1, -1, -2)])
    assert cell(r, 1.5, 1.5) == sm.FREE


def test_gap_fill_is_opt_in_closes_wall_gaps_and_drops_floaters():
    gap_voxel = (2, 0, -4)                                     # ENU x 2..3, north 3..4: the wall gap
    floater = (0, 0, -1)                                       # ENU x 0..1, north 0..1: open floor
    off = build([gap_voxel, floater])
    assert cell(off, 2.5, 3.5) != sm.OCC                       # default: no gap fill
    on = build([gap_voxel, floater], gap_fill=True)
    assert on["stats"]["gap_fill_columns"] == 1                # gap column filled (touches the wall)
    assert cell(on, 2.5, 3.5) == sm.OCC
    assert cell(on, 0.5, 0.5) == sm.FREE                       # floater over seen floor: dropped

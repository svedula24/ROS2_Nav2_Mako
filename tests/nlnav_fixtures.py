"""Shared fixtures for the m20_nlnav logic tests (plain data, no ROS)."""
import numpy as np

from m20_nlnav.world import Grid, Obj, Waypoint, World


def open_grid(w=20.0, h=20.0, res=0.05, origin=(-10.0, -10.0)):
    """All-free cost grid w x h metres."""
    return Grid(np.zeros((int(h / res), int(w / res)), np.int16), origin, res)


def block(grid, x0, y0, x1, y1, cost=100):
    """Paint a rectangle [x0,x1] x [y0,y1] (metres) with a cost."""
    c0, c1 = int((x0 - grid.origin[0]) / grid.res), int((x1 - grid.origin[0]) / grid.res)
    r0, r1 = int((y0 - grid.origin[1]) / grid.res), int((y1 - grid.origin[1]) / grid.res)
    grid.data[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = cost


def office(grid=None):
    objs = [
        Obj(1, "printer", (4.0, 0.0, 0.5), 0.82, 9, (0.6, 0.5, 1.0)),
        Obj(2, "office chair", (2.0, 3.0, 0.5), 0.71, 6, (0.6, 0.6, 1.0)),
        Obj(3, "office chair", (-3.0, 1.0, 0.5), 0.90, 8, (0.6, 0.6, 1.0)),
        Obj(4, "office chair", (6.0, -4.0, 0.5), 0.55, 4, (0.6, 0.6, 1.0)),
        Obj(5, "trash can", (1.0, -2.0, 0.3), 0.66, 5, (0.4, 0.4, 0.6)),
        Obj(6, "filing cabinet", (-5.0, -5.0, 0.8), 0.77, 7, (0.5, 0.6, 1.6)),
        Obj(7, "storage cabinet", (5.0, 5.0, 0.8), 0.61, 3, (1.0, 0.5, 1.8)),
    ]
    wps = {"start": Waypoint("start", 0.0, 0.0, 0.0), "stop3": Waypoint("stop 3", 0.8, -8.1),
           "wp12": Waypoint("WP12", 4.7, 0.5)}
    return World(objects=objs, waypoints=wps, pose=(0.0, 0.0, 0.0),
                 grid=grid if grid is not None else open_grid())

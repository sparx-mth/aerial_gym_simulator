# aerial_gym/env_manager/room_layout.py
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class RoomParams:
    """Per-env static layout: furniture placements etc."""
    furniture: list  # list of dicts: {"x": float, "y": float, "yaw": float}
    keep: int        # number of items used

def quat_from_yaw(yaw: float):
    """Return [qx,qy,qz,qw] for yaw-only rotation."""
    h = 0.5 * yaw
    return [0.0, 0.0, math.sin(h), math.cos(h)]

def sample_room_params(
    rng: np.random.RandomState,
    bmin_xy, bmax_xy,
    keep_range=(4, 10),
    wall_margin=0.35,
    radius_range=(0.25, 0.45),
    max_trials=200,
):
    """
    Sample non-overlapping furniture footprints inside [bmin_xy, bmax_xy].
    Returns RoomParams with XY + yaw for each item.
    """
    xmin, ymin = float(bmin_xy[0]), float(bmin_xy[1])
    xmax, ymax = float(bmax_xy[0]), float(bmax_xy[1])
    xmin += wall_margin; ymin += wall_margin
    xmax -= wall_margin; ymax -= wall_margin

    keep = int(rng.randint(keep_range[0], keep_range[1] + 1))
    placed = []
    radii  = []
    trials = 0

    while len(placed) < keep and trials < max_trials:
        trials += 1
        r  = float(rng.uniform(*radius_range))
        x  = float(rng.uniform(xmin + r, xmax - r))
        y  = float(rng.uniform(ymin + r, ymax - r))
        ok = True
        for (px, py), pr in zip([(p["x"], p["y"]) for p in placed], radii):
            if (x - px)**2 + (y - py)**2 < (r + pr + 0.10)**2:
                ok = False; break
        if ok:
            yaw = float(rng.uniform(-math.pi, math.pi))
            placed.append({"x": x, "y": y, "yaw": yaw})
            radii.append(r)

    return RoomParams(furniture=placed, keep=len(placed))

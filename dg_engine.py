"""
dg_engine.py — Differential growth simulation engine.

Extracted from the Week 2 notebook so both the interactive app and the batch
generator run the *same* code. Nothing in here knows about Streamlit or files;
it only takes parameters and returns node positions.

Ali Chaaraoui & Jay Anupoju — Week 4
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Default parameters. Every length is expressed as a multiple of maxDistance,
# because differential growth is scale-invariant: multiply all positions and
# all lengths by the same factor and the simulation is identical.
# --------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # initial path
    "n_initial": 4,
    "circle_radius": 5.0,

    # lengths (multiples of maxDistance)
    "maxDistance": 1.0,
    "minDistance": 0.5,
    "repulsionRadius": 2.0,
    "brownianRange": 0.15,

    # force strengths (dimensionless)
    "repulsionForce": 0.05,
    "attractionForce": 0.05,
    "alignmentForce": 0.01,

    # rule on/off weights
    "w_repulsion": 1.0,
    "w_attraction": 1.0,
    "w_alignment": 1.0,
    "w_brownian": 1.0,

    # run
    "n_iterations": 160,
    "seed": 42,
}

# Parameters that are swept in batch mode, with sensible ranges.
SWEEPABLE = {
    "n_initial": (3, 12),
    "maxDistance": (0.6, 1.6),
    "minDistance": (0.3, 0.7),
    "repulsionRadius": (1.2, 3.0),
    "brownianRange": (0.02, 0.30),
    "repulsionForce": (0.01, 0.12),
    "attractionForce": (0.01, 0.12),
    "alignmentForce": (0.00, 0.05),
}

# --------------------------------------------------------------------------
# Node ceilings. These answer two different questions and must stay separate.
#
# SIM_MAX_NODES is a resource limit: the point at which a runaway parameter set
# is abandoned so it cannot hang the app. Stopping here says nothing about the
# quality of the shape produced.
#
# VALID_MAX_NODES is a quality judgement: above this a shape is too dense to be
# a useful sample. It also applies to shapes loaded from disk, not just fresh
# runs.
#
# These were previously one value, and run_simulation kept the frame that
# tripped the limit — so the returned shape was over the cap by construction
# and is_valid rejected every capped run automatically.
# --------------------------------------------------------------------------

SIM_MAX_NODES = 6000
VALID_MAX_NODES = 6000


# --------------------------------------------------------------------------
# Initial condition
# --------------------------------------------------------------------------

def initial_ring(n_initial: int, radius: float) -> np.ndarray:
    """n evenly spaced points on a circle, as an (n, 2) array.

    endpoint=False matters: with endpoint=True the last point lands exactly on
    the first, giving a zero-length edge.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_initial), endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


# --------------------------------------------------------------------------
# Topology: subdivision and pruning
# --------------------------------------------------------------------------

def edge_lengths(pts: np.ndarray) -> np.ndarray:
    """Length of edge i, from node i to node i+1, wrapping at the end."""
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def split_edges(pts: np.ndarray, max_distance: float) -> np.ndarray:
    """Insert a midpoint into every edge longer than max_distance.

    This is where growth comes from. Without it nothing new enters the system
    and the loop can only move, never grow.
    """
    lengths = edge_lengths(pts)
    nxt = np.roll(pts, -1, axis=0)
    mids = (pts + nxt) / 2.0

    out = []
    for i in range(len(pts)):
        out.append(pts[i])
        if lengths[i] >= max_distance:
            out.append(mids[i])
    return np.asarray(out, dtype=float)


def prune_nodes(pts: np.ndarray, min_distance: float, floor: int = 4) -> np.ndarray:
    """Drop nodes that have crowded closer than min_distance.

    Measures against the last node *kept*, not the previous node in the input,
    so a whole cluster collapses to a single survivor rather than alternating
    keep-drop-keep.
    """
    if len(pts) <= floor:
        return pts

    kept = [pts[0]]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - kept[-1])) >= min_distance:
            kept.append(pts[i])

    # the closing edge wraps back to node 0 — check it separately
    if len(kept) > floor and np.hypot(*(kept[-1] - kept[0])) < min_distance:
        kept.pop()

    return np.asarray(kept, dtype=float)


# --------------------------------------------------------------------------
# Forces. Each returns an (N, 2) array of displacements rather than moving
# nodes, so `step` can sum them against one frozen snapshot and apply
# everything at once. Moving nodes one at a time makes the result depend on
# the order they happen to be stored in.
# --------------------------------------------------------------------------

def force_repulsion(pts: np.ndarray, p: dict) -> np.ndarray:
    """Push away from every node within repulsionRadius. Closer means stronger."""
    R, F = p["repulsionRadius"], p["repulsionForce"]

    diff = pts[:, None, :] - pts[None, :, :]      # diff[i, j] points from j to i
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)                # a node does not repel itself

    magnitude = np.where(dist < R, (R - dist) * F, 0.0)
    unit = np.divide(
        diff, dist[:, :, None],
        out=np.zeros_like(diff),
        where=np.isfinite(dist)[:, :, None] & (dist[:, :, None] > 0),
    )
    return (unit * magnitude[:, :, None]).sum(axis=1)


def force_attraction(pts: np.ndarray, p: dict) -> np.ndarray:
    """Pull each node toward both of its connected neighbours."""
    F = p["attractionForce"]
    prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
    return F * ((prev - pts) + (nxt - pts))


def force_alignment(pts: np.ndarray, p: dict) -> np.ndarray:
    """Move each node toward the midpoint of its two neighbours (smoothing)."""
    A = p["alignmentForce"]
    prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
    return A * (((prev + nxt) / 2.0) - pts)


def force_brownian(pts: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    """Small random jitter. This is what breaks the symmetry.

    With a perfectly regular polygon under symmetric forces, nothing ever
    changes shape. Brownian motion is what lets one region get slightly ahead,
    stretch, subdivide more, and run away — which is differential growth.
    """
    r = p["brownianRange"]
    return rng.uniform(-r / 2.0, r / 2.0, size=pts.shape)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def step(pts: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    """One iteration: subdivide, prune, then apply all forces at once."""
    pts = split_edges(pts, p["maxDistance"])
    pts = prune_nodes(pts, p["minDistance"])

    d = np.zeros_like(pts)
    if p["w_repulsion"]:
        d += p["w_repulsion"] * force_repulsion(pts, p)
    if p["w_attraction"]:
        d += p["w_attraction"] * force_attraction(pts, p)
    if p["w_alignment"]:
        d += p["w_alignment"] * force_alignment(pts, p)
    if p["w_brownian"]:
        d += p["w_brownian"] * force_brownian(pts, p, rng)

    return pts + d


def run_simulation(p: dict, n_iterations: int | None = None,
                   max_nodes: int = SIM_MAX_NODES, progress=None,
                   info: dict | None = None) -> list[np.ndarray]:
    """Run the simulation, returning a snapshot after every iteration.

    snapshots[0] is the initial ring, so len(snapshots) == iterations_run + 1.

    If the next iteration would push the node count past `max_nodes`, the run
    stops and that frame is discarded. Every frame returned is therefore within
    the cap and usable — the cap ends the run, it does not poison the result.

    Pass a dict as `info` to learn how the run ended; it is filled in with
    `truncated`, `stop_reason`, `iterations_run`, `final_nodes` and `max_nodes`.
    A truncated run is a real shape, just a shorter run than was asked for, so
    the caller decides what to do about it rather than being handed a shape
    that silently fails validation.
    """
    p = {**DEFAULT_PARAMS, **p}
    n = int(n_iterations if n_iterations is not None else p["n_iterations"])
    rng = np.random.default_rng(int(p["seed"]))

    pts = initial_ring(p["n_initial"], p["circle_radius"])
    snapshots = [pts.copy()]
    truncated = False

    for i in range(n):
        nxt = step(pts, p, rng)

        # Test before keeping the frame. Appending first and breaking after
        # returns a shape that is over the cap by construction.
        if len(nxt) > max_nodes:
            truncated = True
            break

        pts = nxt
        snapshots.append(pts.copy())
        if progress is not None:
            progress((i + 1) / n, len(pts))

    if info is not None:
        info.update({
            "truncated": truncated,
            "stop_reason": "node cap" if truncated else "completed",
            "iterations_run": len(snapshots) - 1,
            "iterations_requested": n,
            "final_nodes": int(len(snapshots[-1])),
            "max_nodes": int(max_nodes),
        })

    return snapshots


# --------------------------------------------------------------------------
# Shape descriptors — used for quality filtering and diversity scoring.
# These are deliberately cheap and invariant to where the shape sits, which is
# the same idea as Schnorr's invariant pattern representation.
# --------------------------------------------------------------------------

def resample_closed(pts: np.ndarray, n: int = 256) -> np.ndarray:
    """Resample a closed curve to n points evenly spaced along arc length.

    Every run ends with a different node count. This gives every sample the
    same shape so it can go into an array or a model.
    """
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([
        np.interp(target, s, closed[:, 0]),
        np.interp(target, s, closed[:, 1]),
    ])


def descriptors(pts: np.ndarray) -> dict:
    """A small fixed-length summary of a shape, independent of node count."""
    r = resample_closed(pts, 256)
    centred = r - r.mean(axis=0)
    radii = np.linalg.norm(centred, axis=1)
    perimeter = float(np.linalg.norm(np.diff(np.vstack([r, r[0]]), axis=0), axis=1).sum())

    # signed area via the shoelace formula
    x, y = r[:, 0], r[:, 1]
    area = float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

    # circularity: 1.0 for a perfect circle, lower the more ruffled it is
    circularity = float(4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

    # radial waviness — how many times the radius crosses its own mean
    dev = radii - radii.mean()
    crossings = int(np.sum(np.diff(np.sign(dev)) != 0))

    return {
        "n_nodes": int(len(pts)),
        "perimeter": perimeter,
        "area": area,
        "circularity": circularity,
        "radius_mean": float(radii.mean()),
        "radius_std": float(radii.std()),
        "radius_cv": float(radii.std() / radii.mean()) if radii.mean() > 0 else 0.0,
        "wave_crossings": crossings,
    }


# Descriptor keys used for the diversity comparison between datasets.
DIVERSITY_KEYS = ["circularity", "radius_cv", "wave_crossings", "n_nodes", "perimeter"]


def feature_vector(desc: dict) -> np.ndarray:
    return np.array([desc[k] for k in DIVERSITY_KEYS], dtype=float)


def is_valid(pts: np.ndarray, min_nodes: int = 20, max_nodes: int = VALID_MAX_NODES,
             max_extent: float = 500.0) -> tuple[bool, str]:
    """Reject degenerate runs: barely grew, exploded, or collapsed."""
    if len(pts) < min_nodes:
        return False, f"too few nodes ({len(pts)})"
    if len(pts) > max_nodes:
        return False, f"too many nodes ({len(pts)})"
    extent = float(np.abs(pts - pts.mean(axis=0)).max())
    if not np.isfinite(extent):
        return False, "non-finite coordinates"
    if extent > max_extent:
        return False, f"exploded (extent {extent:.0f})"
    d = descriptors(pts)
    if d["perimeter"] <= 0 or d["area"] <= 0:
        return False, "degenerate shape"
    return True, "ok"

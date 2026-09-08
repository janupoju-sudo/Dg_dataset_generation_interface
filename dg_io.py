"""
dg_io.py — Saving datasets to disk, batch management, and the diversity check.

Datasets are grouped into batches:

    datasets/
        batch_001/
            batch.json               when it was made, how it was made
            dataset_001/
                parameters.json      every parameter value, including the seed
                nodes.csv            iteration, node_id, x, y
                summary.csv          one row per iteration: descriptors
                preview.png          the final shape
                images/iter_000.png  one image per iteration
            dataset_002/
        batch_002/

Everything needed to reproduce a run exactly is in parameters.json. The seed is
stored, so even the Brownian jitter is reproducible.

Ali Chaaraoui & Jay Anupoju — Week 4
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from dg_engine import DIVERSITY_KEYS, descriptors, feature_vector

# Matplotlib without a display, so this works in a script or on a server.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LINE_COLOR = "#1F6F5C"
NODE_COLOR = "#16201B"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render(pts: np.ndarray, path: Path | None = None, size_px: int = 512,
           lim: float | None = None, show_nodes: bool = False, dpi: int = 100):
    """Draw a closed curve. Saves to `path` if given, else returns the figure.

    `lim` fixes the axis limits so every image in a dataset is framed
    identically. Without this, matplotlib autoscales per frame and a model
    trained on the images learns zoom level instead of shape.
    """
    inches = size_px / dpi
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)

    closed = np.vstack([pts, pts[0]])
    ax.plot(closed[:, 0], closed[:, 1], "-", lw=1.1, color=LINE_COLOR)
    if show_nodes and len(pts) <= 400:
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=2.5, color=NODE_COLOR)

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#DCE4DF")

    if lim is not None:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.tight_layout(pad=0.2)

    if path is not None:
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        return None
    return fig


def frame_limit(snapshots: list[np.ndarray], margin: float = 1.08) -> float:
    """One axis limit that fits every frame, centred on the origin."""
    extent = max(float(np.abs(s).max()) for s in snapshots)
    return extent * margin


# --------------------------------------------------------------------------
# Batches
# --------------------------------------------------------------------------

def _next_numbered(parent: Path, prefix: str) -> Path:
    """parent/prefix_001, _002, … — never reuses a number."""
    parent.mkdir(parents=True, exist_ok=True)
    nums = [
        int(p.name.split("_")[-1])
        for p in parent.glob(f"{prefix}_*")
        if p.is_dir() and p.name.split("_")[-1].isdigit()
    ]
    n = max(nums) + 1 if nums else 1
    d = parent / f"{prefix}_{n:03d}"
    d.mkdir()
    return d


def new_batch(root: Path, label: str = "", source: str = "manual") -> Path:
    """Create the next batch folder and write its batch.json."""
    b = _next_numbered(Path(root), "batch")
    (b / "batch.json").write_text(json.dumps({
        "batch": b.name,
        "label": label,
        "source": source,                       # "manual" or "batch run"
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    return b


def list_batches(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.glob("batch_*") if p.is_dir())


def list_datasets(batch_dir: Path) -> list[Path]:
    return sorted(p for p in Path(batch_dir).glob("dataset_*") if p.is_dir())


def list_loose_datasets(root: Path) -> list[Path]:
    """Datasets saved before batching existed, sitting directly under datasets/."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.glob("dataset_*") if p.is_dir())


def batch_meta(batch_dir: Path) -> dict:
    try:
        return json.loads((Path(batch_dir) / "batch.json").read_text())
    except Exception:
        return {"batch": Path(batch_dir).name, "label": "", "source": "?", "created": "?"}


def dir_stats(path: Path) -> tuple[int, float]:
    """(number of image files, total size in MB) for a folder tree."""
    path = Path(path)
    files = list(path.rglob("*"))
    imgs = sum(1 for f in files if f.suffix.lower() == ".png" and f.parent.name == "images")
    size = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    return imgs, size


def delete_batch(batch_dir: Path) -> None:
    """Permanently delete a batch folder and everything inside it.

    This is a real deletion from disk, not a move to Trash. The caller is
    responsible for confirming with the user first.
    """
    batch_dir = Path(batch_dir).resolve()
    if not batch_dir.is_dir():
        raise FileNotFoundError(batch_dir)
    if not (batch_dir.name.startswith("batch_") or batch_dir.name.startswith("dataset_")):
        # refuse to delete anything that isn't ours
        raise ValueError(f"refusing to delete {batch_dir}")
    shutil.rmtree(batch_dir)


def migrate_loose_datasets(root: Path) -> Path | None:
    """Move any pre-batching datasets into a new batch. Returns the batch dir."""
    loose = list_loose_datasets(root)
    if not loose:
        return None
    b = new_batch(root, label="migrated from before batching", source="migrated")
    for i, d in enumerate(loose, start=1):
        d.rename(b / f"dataset_{i:03d}")
    return b


# --------------------------------------------------------------------------
# Saving one dataset
# --------------------------------------------------------------------------

def save_dataset(snapshots: list[np.ndarray], params: dict, batch_dir: Path,
                 save_images: bool = True, image_size: int = 512,
                 note: str = "", progress=None) -> Path:
    """Write one dataset into `batch_dir`. Returns the directory written."""
    batch_dir = Path(batch_dir)
    out = _next_numbered(batch_dir, "dataset")
    lim = frame_limit(snapshots)

    # --- parameters.json -------------------------------------------------
    meta = {
        "batch": batch_dir.name,
        "dataset": out.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "parameters": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in params.items()},
        "n_iterations_run": len(snapshots) - 1,
        "final_node_count": int(len(snapshots[-1])),
        "frame_limit": lim,
        "image_size_px": image_size if save_images else None,
        "final_descriptors": descriptors(snapshots[-1]),
    }
    (out / "parameters.json").write_text(json.dumps(meta, indent=2))

    # --- nodes.csv -------------------------------------------------------
    # Long format. Node counts differ per iteration, so a wide table would be
    # ragged. Iteration 0 is the starting ring.
    with (out / "nodes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "node_id", "x", "y"])
        for it, pts in enumerate(snapshots):
            for nid, (x, y) in enumerate(pts):
                w.writerow([it, nid, f"{x:.6f}", f"{y:.6f}"])

    # --- summary.csv -----------------------------------------------------
    desc_keys = list(descriptors(snapshots[0]).keys())
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration"] + desc_keys)
        for it, pts in enumerate(snapshots):
            d = descriptors(pts)
            w.writerow([it] + [f"{d[k]:.6f}" if isinstance(d[k], float) else d[k]
                               for k in desc_keys])

    # --- images ----------------------------------------------------------
    if save_images:
        img_dir = out / "images"
        img_dir.mkdir()
        for it, pts in enumerate(snapshots):
            render(pts, img_dir / f"iter_{it:03d}.png", size_px=image_size, lim=lim)
            if progress is not None:
                progress((it + 1) / len(snapshots))

    render(snapshots[-1], out / "preview.png", size_px=512, lim=lim)
    return out


# --------------------------------------------------------------------------
# Diversity — "only save if it differs from existing ones"
# --------------------------------------------------------------------------

def load_features(dataset_dirs: list[Path]) -> list[tuple[str, np.ndarray]]:
    """Feature vector of each given dataset."""
    out = []
    for d in dataset_dirs:
        try:
            meta = json.loads((Path(d) / "parameters.json").read_text())
            out.append((meta["dataset"], feature_vector(meta["final_descriptors"])))
        except Exception:
            continue
    return out


def _scales(vectors: list[np.ndarray]) -> np.ndarray:
    """Per-feature scale, so features with big units don't dominate."""
    if not vectors:
        return np.ones(len(DIVERSITY_KEYS))
    arr = np.vstack(vectors)
    s = arr.std(axis=0)
    s[s == 0] = 1.0
    return s


def diversity_check(candidate_desc: dict, dataset_dirs: list[Path],
                    threshold: float = 0.35) -> tuple[bool, float, str]:
    """Is this shape different enough from the given datasets?

    Returns (is_novel, distance_to_nearest, name_of_nearest).
    Distance is normalised, so `threshold` is roughly "fraction of a standard
    deviation across the compared collection".

    Pass only the current batch's datasets to keep batches independent, or
    every dataset to enforce novelty across the whole library.
    """
    existing = load_features(list(dataset_dirs))
    if not existing:
        return True, float("inf"), ""

    cand = feature_vector(candidate_desc)
    vecs = [v for _, v in existing]
    scale = _scales(vecs + [cand])

    dists = [(float(np.linalg.norm((cand - v) / scale)), name) for name, v in existing]
    nearest_d, nearest_name = min(dists)
    return nearest_d >= threshold, nearest_d, nearest_name


def all_datasets(root: Path) -> list[Path]:
    """Every dataset across every batch."""
    out = []
    for b in list_batches(root):
        out.extend(list_datasets(b))
    return out

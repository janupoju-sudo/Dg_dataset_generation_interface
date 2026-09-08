"""
dg_app.py — Differential Growth Explorer.

Interactive interface for generating differential growth datasets.

    pip install streamlit numpy matplotlib
    streamlit run dg_app.py

Three tabs:
  Explore   sliders, live preview, save a dataset into a batch
  Batch     sweep parameter ranges automatically, each run makes a new batch
  Library   browse batches, delete a whole batch

Ali Chaaraoui & Jay Anupoju — Week 4
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from dg_engine import (
    DEFAULT_PARAMS, SWEEPABLE, descriptors, is_valid, run_simulation,
)
from dg_io import (
    all_datasets, batch_meta, delete_batch, dir_stats, diversity_check,
    list_batches, list_datasets, list_loose_datasets, migrate_loose_datasets,
    new_batch, render, save_dataset,
)

DATA_ROOT = Path(__file__).parent / "datasets"

st.set_page_config(page_title="Differential Growth Explorer", layout="wide")


# ==========================================================================
# Sidebar — parameters
# ==========================================================================

def sidebar_params() -> dict:
    st.sidebar.title("Parameters")

    st.sidebar.caption(
        "Lengths are multiples of maxDistance. Differential growth is "
        "scale-invariant, so this tuning transfers to any circle radius."
    )

    st.sidebar.subheader("Initial path")
    n_initial = st.sidebar.slider("Starting nodes", 3, 16, 4, 1)
    circle_radius = st.sidebar.slider("Circle radius", 1.0, 20.0, 5.0, 0.5)

    st.sidebar.subheader("Lengths")
    maxDistance = st.sidebar.slider("maxDistance  (split above this)", 0.2, 3.0, 1.0, 0.05)
    minDistance = st.sidebar.slider("minDistance  (prune below this)", 0.05, 1.5, 0.5, 0.05)
    repulsionRadius = st.sidebar.slider("repulsionRadius", 0.5, 5.0, 2.0, 0.1)
    brownianRange = st.sidebar.slider("brownianRange", 0.0, 0.6, 0.15, 0.01)

    if minDistance > 0.5 * maxDistance:
        st.sidebar.warning(
            f"minDistance ({minDistance:.2f}) is more than half of maxDistance "
            f"({maxDistance:.2f}). Splitting an edge creates two halves of about "
            f"{maxDistance/2:.2f}, which pruning will immediately delete — "
            "split and prune will fight each other."
        )

    st.sidebar.subheader("Force strengths")
    repulsionForce = st.sidebar.slider("repulsionForce", 0.0, 0.3, 0.05, 0.005)
    attractionForce = st.sidebar.slider("attractionForce", 0.0, 0.3, 0.05, 0.005)
    alignmentForce = st.sidebar.slider("alignmentForce", 0.0, 0.2, 0.01, 0.005)

    st.sidebar.subheader("Rules on / off")
    c1, c2 = st.sidebar.columns(2)
    w_repulsion = 1.0 if c1.checkbox("Repulsion", True) else 0.0
    w_attraction = 1.0 if c2.checkbox("Attraction", True) else 0.0
    w_alignment = 1.0 if c1.checkbox("Alignment", True) else 0.0
    w_brownian = 1.0 if c2.checkbox("Brownian", True) else 0.0

    if w_brownian == 0.0:
        st.sidebar.info(
            "With Brownian off the shape stays symmetric and growth stalls — "
            "useful for the ablation figures, not for datasets."
        )

    st.sidebar.subheader("Run")
    n_iterations = st.sidebar.slider("Iterations", 10, 400, 160, 10)
    seed = st.sidebar.number_input("Random seed", 0, 999_999, 42, 1)

    return {
        "n_initial": n_initial,
        "circle_radius": circle_radius,
        "maxDistance": maxDistance,
        "minDistance": minDistance,
        "repulsionRadius": repulsionRadius,
        "brownianRange": brownianRange,
        "repulsionForce": repulsionForce,
        "attractionForce": attractionForce,
        "alignmentForce": alignmentForce,
        "w_repulsion": w_repulsion,
        "w_attraction": w_attraction,
        "w_alignment": w_alignment,
        "w_brownian": w_brownian,
        "n_iterations": n_iterations,
        "seed": int(seed),
    }


# ==========================================================================
# Shared helpers
# ==========================================================================

def compare_scope(label_key: str) -> tuple[str, list[Path]]:
    """Which datasets the novelty check measures against."""
    scope = st.radio(
        "Compare novelty against",
        ["This batch only", "Every batch"],
        horizontal=True, key=label_key,
        help="'This batch only' keeps batches independent, so a rerun isn't "
             "blocked by shapes in an older batch.",
    )
    return scope, []


# ==========================================================================
# Tab 1 — Explore
# ==========================================================================

def tab_explore(params: dict):
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Preview")
        if st.button("Run simulation", type="primary", use_container_width=True):
            bar = st.progress(0.0, text="Running…")

            def prog(frac, n_nodes):
                bar.progress(frac, text=f"Iteration progress — {n_nodes} nodes")

            st.session_state["snapshots"] = run_simulation(params, progress=prog)
            st.session_state["params"] = dict(params)
            bar.empty()

        snaps = st.session_state.get("snapshots")
        if snaps is None:
            st.info("Set parameters in the sidebar, then press **Run simulation**.")
            return

        show_nodes = st.checkbox("Show individual nodes", value=False)
        it = st.slider("Iteration", 0, len(snaps) - 1, len(snaps) - 1)
        lim = max(float(np.abs(s).max()) for s in snaps) * 1.08
        st.pyplot(render(snaps[it], lim=lim, show_nodes=show_nodes), use_container_width=True)

    with right:
        snaps = st.session_state.get("snapshots")
        if snaps is None:
            return

        final = snaps[-1]
        d = descriptors(final)
        ok, reason = is_valid(final)

        st.subheader("This run")
        if not ok:
            st.error(f"Rejected: {reason}")
        else:
            st.success("Valid shape")

        a, b = st.columns(2)
        a.metric("Final nodes", d["n_nodes"])
        b.metric("Iterations", len(snaps) - 1)
        a.metric("Circularity", f"{d['circularity']:.3f}",
                 help="1.0 is a perfect circle; lower means more ruffled")
        b.metric("Waviness", d["wave_crossings"],
                 help="How many times the radius crosses its own mean")
        a.metric("Perimeter", f"{d['perimeter']:.1f}")
        b.metric("Radius CV", f"{d['radius_cv']:.3f}")

        st.divider()
        st.subheader("Save into a batch")

        batches = list_batches(DATA_ROOT)
        choices = ["➕ New batch"] + [p.name for p in reversed(batches)]
        picked = st.selectbox("Batch", choices)

        new_label = ""
        if picked == "➕ New batch":
            new_label = st.text_input("Batch label (optional)",
                                      placeholder="e.g. manual picks, 8-node starts")
            target = None
        else:
            target = DATA_ROOT / picked

        scope = st.radio("Compare novelty against", ["This batch only", "Every batch"],
                         horizontal=True, key="scope_explore")
        if target is None:
            compare_dirs = [] if scope == "This batch only" else all_datasets(DATA_ROOT)
        else:
            compare_dirs = list_datasets(target) if scope == "This batch only" \
                else all_datasets(DATA_ROOT)

        threshold = st.slider(
            "Novelty threshold", 0.0, 2.0, 0.35, 0.05,
            help="How different this must be from the compared datasets.",
        )
        novel, dist, nearest = diversity_check(d, compare_dirs, threshold)

        if not compare_dirs:
            st.info("Nothing to compare against yet — this will be the first.")
        elif novel:
            st.success(f"Novel — distance {dist:.2f} from nearest ({nearest})")
        else:
            st.warning(
                f"Similar to **{nearest}** (distance {dist:.2f}, below {threshold:.2f}). "
                "Change the parameters, or override below."
            )

        note = st.text_input("Note (optional)", placeholder="e.g. tight folds, 8 start nodes")
        save_images = st.checkbox("Save an image for every iteration", value=True)
        image_size = st.select_slider("Image size (px)", [128, 256, 512], value=512)
        override = st.checkbox("Save even if not novel", value=False)

        if st.button("Save dataset", type="primary",
                     disabled=not (ok and (novel or override)),
                     use_container_width=True):
            batch_dir = target or new_batch(DATA_ROOT, label=new_label, source="manual")
            bar = st.progress(0.0, text="Writing…")
            out = save_dataset(
                snaps, st.session_state["params"], batch_dir,
                save_images=save_images, image_size=image_size, note=note,
                progress=lambda f: bar.progress(f, text="Writing images…"),
            )
            bar.empty()
            st.success(f"Saved to `{batch_dir.name}/{out.name}`")


# ==========================================================================
# Tab 2 — Batch
# ==========================================================================

def tab_batch(base_params: dict):
    st.subheader("Batch generation")
    st.caption(
        "Each run creates a new batch folder. Samples the parameter space, "
        "discards degenerate runs, and keeps only shapes that differ enough. "
        "Settle the ranges by hand in Explore first."
    )

    st.markdown("**Which parameters to vary**")
    chosen, ranges = [], {}
    cols = st.columns(2)
    for i, (name, (lo, hi)) in enumerate(SWEEPABLE.items()):
        with cols[i % 2]:
            on = st.checkbox(name, value=name in
                             ("maxDistance", "repulsionRadius", "brownianRange", "repulsionForce"),
                             key=f"sw_{name}")
            if on:
                chosen.append(name)
                if name == "n_initial":
                    ranges[name] = st.slider(" ", int(lo), int(hi), (3, 10),
                                             key=f"rg_{name}", label_visibility="collapsed")
                else:
                    ranges[name] = st.slider(" ", float(lo), float(hi), (float(lo), float(hi)),
                                             key=f"rg_{name}", label_visibility="collapsed")

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    n_trials = c1.number_input("Trials to attempt", 5, 2000, 50, 5)
    target = c2.number_input("Stop after N kept", 1, 500, 10, 1)
    threshold = c3.slider("Novelty threshold", 0.0, 2.0, 0.35, 0.05)
    n_iterations = c4.number_input("Iterations per run", 10, 400, 160, 10)

    label = st.text_input("Batch label (optional)", placeholder="e.g. wide repulsion sweep")
    scope = st.radio("Compare novelty against", ["This batch only", "Every batch"],
                     horizontal=True, key="scope_batch",
                     help="'This batch only' means a rerun starts fresh and isn't "
                          "blocked by shapes in older batches.")

    save_images = st.checkbox("Save images for each kept dataset", value=True)
    image_size = st.select_slider("Image size (px)", [128, 256, 512], value=256, key="batch_px")
    seed_mode = st.radio("Seed", ["New seed per trial", "Fixed seed (vary parameters only)"],
                         horizontal=True)

    if not chosen:
        st.warning("Select at least one parameter to vary.")
        return

    if st.button("Run batch", type="primary"):
        batch_dir = new_batch(DATA_ROOT, label=label, source="batch run")
        st.caption(f"Writing into **{batch_dir.name}**")

        rng = np.random.default_rng()
        bar = st.progress(0.0)
        log_box = st.empty()
        log, kept, t = [], 0, 0

        for t in range(int(n_trials)):
            p = dict(base_params)
            p["n_iterations"] = int(n_iterations)
            p["seed"] = int(rng.integers(0, 1_000_000)) if seed_mode.startswith("New") \
                else int(base_params["seed"])

            for name in chosen:
                lo, hi = ranges[name]
                p[name] = int(rng.integers(lo, hi + 1)) if name == "n_initial" \
                    else float(rng.uniform(lo, hi))

            try:
                snaps = run_simulation(p, max_nodes=6000)
            except Exception as exc:
                log.append(f"trial {t+1}: error — {exc}")
                continue

            final = snaps[-1]
            ok, reason = is_valid(final)
            if not ok:
                log.append(f"trial {t+1}: rejected — {reason}")
            else:
                d = descriptors(final)
                compare_dirs = list_datasets(batch_dir) if scope == "This batch only" \
                    else all_datasets(DATA_ROOT)
                novel, dist, nearest = diversity_check(d, compare_dirs, threshold)
                if not novel:
                    log.append(f"trial {t+1}: too similar to {nearest} (d={dist:.2f})")
                else:
                    out = save_dataset(snaps, p, batch_dir, save_images=save_images,
                                       image_size=image_size, note=f"trial {t+1}")
                    kept += 1
                    log.append(f"trial {t+1}: **kept → {out.name}** "
                               f"({d['n_nodes']} nodes, circularity {d['circularity']:.2f})")

            bar.progress((t + 1) / n_trials, text=f"{t+1}/{n_trials} · {kept} kept")
            log_box.markdown("\n\n".join(f"- {line}" for line in log[-12:]))

            if kept >= target:
                break

        bar.empty()
        if kept == 0:
            delete_batch(batch_dir)
            st.error("Nothing kept — the empty batch was removed. Widen the ranges "
                     "or lower the novelty threshold.")
        else:
            st.success(f"{batch_dir.name} — {kept} datasets saved from {t+1} trials.")
            if kept < target:
                st.info(
                    "Fewer kept than requested. Widen the ranges, lower the novelty "
                    "threshold, or raise the trial count — a high rejection rate "
                    "usually means the ranges produce similar shapes."
                )


# ==========================================================================
# Tab 3 — Library
# ==========================================================================

def render_dataset_card(d: Path):
    st.caption(f"**{d.name}**")

    try:
        meta = json.loads((d / "parameters.json").read_text())
    except Exception:
        meta = None

    if meta:
        with st.expander("parameters"):
            st.json(meta["parameters"])
    else:
        st.caption("(could not read parameters.json)")

    preview = d / "preview.png"
    if preview.exists():
        st.image(str(preview), use_container_width=True)

    if meta:
        fd = meta["final_descriptors"]
        st.caption(f"{fd['n_nodes']} nodes · circ {fd['circularity']:.2f} · "
                   f"{meta['n_iterations_run']} iters")
        if meta.get("note"):
            st.caption(f"_{meta['note']}_")


def delete_controls(path: Path, key: str, what: str):
    """Two-step delete. Nothing is removed until the second press."""
    pending = st.session_state.get("pending_delete")

    if pending != str(path):
        if st.button("🗑 Delete", key=f"del_{key}", use_container_width=True):
            st.session_state["pending_delete"] = str(path)
            st.rerun()
        return

    st.warning(
        f"Permanently delete **{what}** and everything inside it? "
        "This removes the folder from your computer — it does not go to the Trash "
        "and cannot be undone."
    )
    c1, c2 = st.columns(2)
    if c1.button("Yes, delete", key=f"yes_{key}", type="primary", use_container_width=True):
        delete_batch(path)
        st.session_state["pending_delete"] = None
        st.rerun()
    if c2.button("Cancel", key=f"no_{key}", use_container_width=True):
        st.session_state["pending_delete"] = None
        st.rerun()


def tab_library():
    st.subheader("Saved batches")

    loose = list_loose_datasets(DATA_ROOT)
    if loose:
        st.info(f"{len(loose)} dataset(s) sit outside any batch, from before batching existed.")
        if st.button("Move them into a new batch"):
            b = migrate_loose_datasets(DATA_ROOT)
            st.success(f"Moved into {b.name}")
            st.rerun()

    batches = list_batches(DATA_ROOT)
    if not batches:
        st.info("No batches yet. Save one from Explore, or run the Batch tab.")
        return

    n_ds = sum(len(list_datasets(b)) for b in batches)
    imgs, size = dir_stats(DATA_ROOT)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batches", len(batches))
    c2.metric("Datasets", n_ds)
    c3.metric("Images", imgs)
    c4.metric("Total size", f"{size:.1f} MB")

    st.divider()

    for b in reversed(batches):
        meta = batch_meta(b)
        ds = list_datasets(b)
        b_imgs, b_size = dir_stats(b)

        title = f"{b.name} — {len(ds)} datasets · {b_imgs} images · {b_size:.1f} MB"
        if meta.get("label"):
            title += f" · {meta['label']}"

        with st.expander(title, expanded=(b == batches[-1])):
            head, ctrl = st.columns([4, 1])
            head.caption(f"{meta.get('source', '?')} · created {meta.get('created', '?')}")
            with ctrl:
                delete_controls(b, key=b.name, what=b.name)

            if not ds:
                st.caption("(empty)")
                continue

            cols = st.columns(4)
            for i, d in enumerate(ds):
                with cols[i % 4]:
                    render_dataset_card(d)


# ==========================================================================

st.title("Differential Growth Explorer")
st.caption("Week 4 · dataset generation for the inverse-design project")

params = sidebar_params()
t1, t2, t3 = st.tabs(["Explore", "Batch", "Library"])
with t1:
    tab_explore(params)
with t2:
    tab_batch(params)
with t3:
    tab_library()

"""
06_phase_analysis.py
────────────────────────────────────────────────────────────────────────────────
Phase-locked per-position accuracy.

05_evaluate.py bins tokens by `absolute_position % 72`.  Real S. islandicus
spacers vary 30-45 nt, so the true period drifts along an array; over a 512 nt
window (~7 periods) a 2-4 nt per-period drift accumulates to 14-28 nt and
smears any phase-locked signal into a flat line.  That figure therefore cannot
distinguish "no position dependence" from "position dependence destroyed by
the binning".

This script derives phase from the ground-truth labels instead:

  Panel A  Offset from the start of each token's own repeat block.
           Phase 0..35 is inside the repeat, 36+ is the following spacer.
           Phase-locked by construction — no drift.

  Panel B  Signed distance to the nearest repeat/spacer boundary.
           Negative = inside spacer, positive = inside repeat.
           Shows whether errors concentrate at block edges.

Operates on the test predictions already saved by 04_train.py.  No retraining.

Usage:
    python 06_phase_analysis.py --tag final
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PE_ORDER  = ["learned", "sinusoidal", "rope", "alibi"]
PE_LABELS = {"learned": "Learned Abs.", "sinusoidal": "Sinusoidal",
             "rope": "RoPE",            "alibi": "ALiBi"}
PE_COLORS = {"learned": "#6C757D", "sinusoidal": "#0D6EFD",
             "rope":    "#198754", "alibi":      "#E0A800"}
SEEDS = [0, 1, 2, 3, 4, 5]

MAX_PHASE   = 80   # nt of phase axis to plot (one period plus margin)
EDGE_WINDOW = 25   # nt either side of a boundary for panel B


def split_runs(pos: np.ndarray) -> list[tuple[int, int]]:
    """
    Split the concatenated prediction stream into contiguous segments.
    A segment break is any place where the absolute position does not
    increment by exactly 1 (i.e. a window or seed boundary).
    """
    if len(pos) == 0:
        return []
    breaks = np.nonzero(np.diff(pos) != 1)[0] + 1
    edges  = np.concatenate([[0], breaks, [len(pos)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1]


def phase_from_labels(labels: np.ndarray) -> np.ndarray:
    """
    For each token, the offset from the start of its own repeat block.

    Walks the label sequence; every 0->1 transition resets the counter.
    Tokens before the first repeat get -1 (excluded downstream).
    """
    phase   = np.full(len(labels), -1, dtype=np.int64)
    counter = -1
    for i, lab in enumerate(labels):
        if lab == 1 and (i == 0 or labels[i - 1] == 0):
            counter = 0           # start of a new repeat block
        elif counter >= 0:
            counter += 1
        phase[i] = counter
    return phase


def signed_edge_distance(labels: np.ndarray) -> np.ndarray:
    """
    Signed distance to the nearest repeat/spacer boundary.
    Positive = inside a repeat, negative = inside a spacer.
    Magnitude = nt to the closest boundary.
    """
    n = len(labels)
    # Distance to the nearest index where the label changes
    change = np.nonzero(np.diff(labels) != 0)[0]        # boundary between i, i+1
    if len(change) == 0:
        return np.zeros(n, dtype=np.int64)
    idx  = np.arange(n)
    # nearest boundary position (use the midpoint between the two tokens)
    bpos = change + 0.5
    dist = np.min(np.abs(idx[:, None] - bpos[None, :]), axis=1)
    sign = np.where(labels == 1, 1, -1)
    return (np.ceil(dist) * sign).astype(np.int64)


def accumulate(pe: str, tag: str):
    """
    Returns (phase_acc, phase_n, edge_acc, edge_n) accumulated over all seeds.
    """
    root = Path(f"outputs/checkpoints_{tag}")
    p_hit = np.zeros(MAX_PHASE); p_tot = np.zeros(MAX_PHASE)
    e_rng = 2 * EDGE_WINDOW + 1
    e_hit = np.zeros(e_rng);     e_tot = np.zeros(e_rng)

    found = False
    for s in SEEDS:
        f = root / pe / f"seed{s}" / "test_preds.npz"
        if not f.exists():
            continue
        found = True
        d = np.load(f)
        preds, labels, pos = d["preds"], d["labels"], d["pos"]

        for a, b in split_runs(pos):
            L   = labels[a:b]
            P   = preds[a:b]
            ok  = (P == L).astype(np.float64)

            ph  = phase_from_labels(L)
            m   = (ph >= 0) & (ph < MAX_PHASE)
            np.add.at(p_hit, ph[m], ok[m])
            np.add.at(p_tot, ph[m], 1.0)

            ed  = signed_edge_distance(L)
            m2  = np.abs(ed) <= EDGE_WINDOW
            np.add.at(e_hit, ed[m2] + EDGE_WINDOW, ok[m2])
            np.add.at(e_tot, ed[m2] + EDGE_WINDOW, 1.0)

    if not found:
        return None
    return (p_hit / np.maximum(p_tot, 1), p_tot,
            e_hit / np.maximum(e_tot, 1), e_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="final")
    args = ap.parse_args()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    any_data = False
    summary  = []

    for pe in PE_ORDER:
        res = accumulate(pe, args.tag)
        if res is None:
            continue
        any_data = True
        p_acc, p_tot, e_acc, e_tot = res

        valid = p_tot > 200
        axA.plot(np.arange(MAX_PHASE)[valid], p_acc[valid],
                 lw=2.0, alpha=0.9, color=PE_COLORS[pe], label=PE_LABELS[pe])

        x = np.arange(-EDGE_WINDOW, EDGE_WINDOW + 1)
        ev = e_tot > 200
        axB.plot(x[ev], e_acc[ev], lw=2.0, alpha=0.9,
                 color=PE_COLORS[pe], label=PE_LABELS[pe])

        # Quantify how much accuracy actually varies with phase
        a = p_acc[valid]
        summary.append((pe, float(a.min()), float(a.max()),
                        float(a.max() - a.min()), float(a.std())))

    if not any_data:
        print(f"[ERROR] No test_preds found under outputs/checkpoints_{args.tag}/")
        return

    axA.axvline(36, color="gray", ls="--", lw=1.0, alpha=0.8)
    axA.text(18, 0.26, "repeat", ha="center", fontsize=9, color="#444")
    axA.text(58, 0.26, "spacer", ha="center", fontsize=9, color="#444")
    axA.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.7)
    axA.set_xlabel("Phase: nt from start of repeat block", fontsize=11)
    axA.set_ylabel("Token accuracy", fontsize=11)
    axA.set_title("A. Phase-locked accuracy (derived from labels)", fontsize=11)
    axA.set_ylim(0.2, 1.0)
    axA.legend(fontsize=9, loc="lower right", framealpha=0.9)
    axA.spines[["top", "right"]].set_visible(False)
    axA.yaxis.grid(True, ls="--", alpha=0.35)

    axB.axvline(0, color="gray", ls="--", lw=1.0, alpha=0.8)
    axB.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.7)
    axB.set_xlabel("Signed distance to nearest boundary (nt)\n"
                   "negative = inside spacer,  positive = inside repeat",
                   fontsize=10)
    axB.set_ylabel("Token accuracy", fontsize=11)
    axB.set_title("B. Accuracy vs distance from block boundary", fontsize=11)
    axB.set_ylim(0.2, 1.0)
    axB.legend(fontsize=9, loc="lower right", framealpha=0.9)
    axB.spines[["top", "right"]].set_visible(False)
    axB.yaxis.grid(True, ls="--", alpha=0.35)

    fig.suptitle(f"Phase-locked error structure, S. islandicus CRISPR arrays "
                 f"({args.tag})", fontsize=12, y=1.03)
    plt.tight_layout()

    out_dir = Path(f"outputs/figures_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "phase_locked.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n── Phase-dependence of accuracy (panel A) ──")
    print(f"  {'PE type':<14}{'min':>8}{'max':>8}{'range':>9}{'std':>8}   interpretation")
    print("  " + "-" * 72)
    for pe, lo, hi, rng, sd in summary:
        interp = ("strong phase dependence" if rng > 0.25 else
                  "moderate phase dependence" if rng > 0.12 else
                  "weak / none")
        print(f"  {PE_LABELS[pe]:<14}{lo:>8.3f}{hi:>8.3f}{rng:>9.3f}{sd:>8.3f}   {interp}")

    print(f"\n  A flat line here means accuracy does not depend on where in the")
    print(f"  repeat-spacer cycle a token sits. Unlike the modulo-72 figure,")
    print(f"  this binning is drift-free, so flat is now an interpretable result.")
    print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()

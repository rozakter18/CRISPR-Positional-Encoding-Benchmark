"""
08_boundary_lag.py
────────────────────────────────────────────────────────────────────────────────
Measures how far a model's predicted block boundaries lag the true ones.

07_error_concentration.py bins by UNSIGNED distance to the nearest boundary, so
it averages the first nucleotides of a block together with the last ones.  If a
model is late to switch classes after a transition -- accurate at block ends,
wrong at block starts -- that averaging cancels most of the signal and reports
a weak enrichment.

This script uses a signed coordinate instead:

    offset = nt elapsed since the most recent label transition
             (0 = first nucleotide of a new block)

Accuracy as a function of offset separates the two cases.  A model that tracks
boundaries exactly is flat in offset.  A model that keeps predicting the
previous class for k nt shows a trough over offset 0..k and a plateau after.

Reported per positional encoding, and separately for repeat and spacer blocks:

    floor      accuracy at offset 0 (first nt of a new block)
    plateau    mean accuracy at offset >= PLATEAU_FROM
    L50        smallest offset where accuracy reaches halfway from floor
               to plateau -- the effective lag, in nt
    err<L50    share of all errors occurring before that recovery point

Reads the test predictions saved by 04_train.py.  No retraining.

Usage:
    python 08_boundary_lag.py --tag final
    python 08_boundary_lag.py --tag final --latex
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

MAX_OFFSET   = 36    # nt of offset axis (one block)
PLATEAU_FROM = 20    # offsets >= this define the plateau
MIN_COUNT    = 200   # minimum tokens in a bin before it is plotted


def split_runs(pos: np.ndarray) -> list[tuple[int, int]]:
    if len(pos) == 0:
        return []
    breaks = np.nonzero(np.diff(pos) != 1)[0] + 1
    edges  = np.concatenate([[0], breaks, [len(pos)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1]


def offset_since_transition(labels: np.ndarray) -> np.ndarray:
    """
    nt elapsed since the most recent label change.
    Tokens before the first transition in a segment get -1 (block start unknown).
    Vectorised: offset = index - index_of_last_transition.
    """
    n = len(labels)
    is_start = np.zeros(n, dtype=bool)
    is_start[1:] = labels[1:] != labels[:-1]
    idx = np.arange(n)
    last_start = np.maximum.accumulate(np.where(is_start, idx, -1))
    out = idx - last_start
    out[last_start < 0] = -1          # no transition seen yet in this segment
    return out


def accumulate(pe: str, tag: str):
    """
    Returns dict with per-offset accuracy for all tokens, repeat blocks only,
    and spacer blocks only; plus total error count.
    """
    root = Path(f"outputs/checkpoints_{tag}")
    hit = {k: np.zeros(MAX_OFFSET) for k in ("all", "repeat", "spacer")}
    tot = {k: np.zeros(MAX_OFFSET) for k in ("all", "repeat", "spacer")}
    n_err = 0
    found = False

    for s in SEEDS:
        f = root / pe / f"seed{s}" / "test_preds.npz"
        if not f.exists():
            continue
        found = True
        d = np.load(f)
        preds, labels, pos = d["preds"], d["labels"], d["pos"]

        for a, b in split_runs(pos):
            L  = labels[a:b]
            ok = (preds[a:b] == L).astype(np.float64)
            off = offset_since_transition(L)
            m = (off >= 0) & (off < MAX_OFFSET)
            n_err += int((1 - ok).sum())

            np.add.at(hit["all"], off[m], ok[m])
            np.add.at(tot["all"], off[m], 1.0)
            for key, val in (("repeat", 1), ("spacer", 0)):
                mm = m & (L == val)
                np.add.at(hit[key], off[mm], ok[mm])
                np.add.at(tot[key], off[mm], 1.0)

    if not found:
        return None
    acc = {k: hit[k] / np.maximum(tot[k], 1) for k in hit}
    return acc, tot, n_err


def describe(acc: np.ndarray, tot: np.ndarray, n_err: int) -> dict:
    """floor / plateau / L50 / share of errors before L50."""
    valid = tot > MIN_COUNT
    if not valid.any():
        return {}
    plateau_mask = valid.copy()
    plateau_mask[:PLATEAU_FROM] = False
    plateau = float(acc[plateau_mask].mean()) if plateau_mask.any() else float("nan")
    floor   = float(acc[0]) if valid[0] else float("nan")

    half = (floor + plateau) / 2.0
    L50 = next((k for k in range(MAX_OFFSET)
                if valid[k] and acc[k] >= half), MAX_OFFSET)

    err_before = float(((1 - acc[:L50]) * tot[:L50]).sum())
    return {"floor": floor, "plateau": plateau, "L50": L50,
            "err_frac": err_before / max(n_err, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="final")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    results = {}
    for pe in PE_ORDER:
        r = accumulate(pe, args.tag)
        if r is not None:
            results[pe] = r
    if not results:
        print(f"[ERROR] No test_preds under outputs/checkpoints_{args.tag}/")
        return

    print(f"\n══ Boundary lag ({args.tag}) ══\n")
    print("  offset = nt since the most recent label transition")
    print("  A model that tracks boundaries exactly is flat in offset.")
    print("  L50 = nt needed to recover halfway to plateau  ->  effective lag.\n")

    for scope, title in (("all", "all blocks"),
                         ("repeat", "repeat blocks"),
                         ("spacer", "spacer blocks")):
        print(f"  ── {title} " + "─" * (56 - len(title)))
        print(f"  {'PE type':<14}{'floor':>8}{'plateau':>9}{'L50 (nt)':>10}"
              f"{'err<L50':>9}")
        for pe, (acc, tot, n_err) in results.items():
            d = describe(acc[scope], tot[scope], n_err)
            if not d:
                continue
            print(f"  {PE_LABELS[pe]:<14}{d['floor']:>8.3f}{d['plateau']:>9.3f}"
                  f"{d['L50']:>10d}{100*d['err_frac']:>8.0f}%")
        print()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, scope, title in zip(axes,
                                ("all", "repeat", "spacer"),
                                ("All blocks", "Repeat blocks", "Spacer blocks")):
        for pe, (acc, tot, _n) in results.items():
            v = tot[scope] > MIN_COUNT
            ax.plot(np.arange(MAX_OFFSET)[v], acc[scope][v],
                    lw=2.0, alpha=0.9, color=PE_COLORS[pe], label=PE_LABELS[pe])
        ax.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel("nt since last transition", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, MAX_OFFSET - 1)
        ax.set_ylim(0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, ls="--", alpha=0.35)
    axes[0].set_ylabel("Token accuracy", fontsize=11)
    axes[0].legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.suptitle("Accuracy recovery after a block transition "
                 f"({args.tag}) — a flat line means boundaries are tracked exactly",
                 fontsize=12, y=1.04)
    plt.tight_layout()

    out_dir = Path(f"outputs/figures_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "boundary_lag.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")

    if args.latex:
        tex = out_dir / "boundary_lag.tex"
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{Accuracy recovery after a repeat/spacer transition. "
             r"Offset is nucleotides elapsed since the most recent boundary; "
             r"floor is accuracy at the first nucleotide of a new block and "
             r"plateau the mean accuracy at offsets $\geq 20$ nt. $L_{50}$ is "
             r"the offset at which accuracy reaches the midpoint between them, "
             r"i.e. the effective lag of the predicted boundary. A model that "
             r"localised boundaries exactly would have floor $=$ plateau and "
             r"$L_{50}=0$.}",
             r"\label{tab:boundary-lag}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Positional encoding & Floor & Plateau & $L_{50}$ (nt) & "
             r"Errors before $L_{50}$ \\", r"\midrule"]
        for pe, (acc, tot, n_err) in results.items():
            d = describe(acc["all"], tot["all"], n_err)
            L.append(f"{PE_LABELS[pe]} & {d['floor']:.3f} & {d['plateau']:.3f} & "
                     f"{d['L50']} & {100*d['err_frac']:.0f}\\% \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tex.write_text("\n".join(L))
        print(f"  saved → {tex}")


if __name__ == "__main__":
    main()

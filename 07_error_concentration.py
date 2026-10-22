"""
07_error_concentration.py
────────────────────────────────────────────────────────────────────────────────
Quantifies the claim that classification errors concentrate at repeat/spacer
boundaries rather than being spread across the array.

For each positional encoding it reports, at several boundary-zone widths:

    tokens%     share of test tokens lying within +/-N nt of a boundary
    errors%     share of all misclassified tokens lying in that zone
    enrich      errors% / tokens%   (1.0 = errors spread uniformly)
    acc_zone    accuracy inside the boundary zone
    acc_int     accuracy in block interiors (outside the zone)

A large enrichment with high interior accuracy means the model recognises
repeat vs spacer composition but cannot resolve where one block ends and the
next begins -- i.e. it is not exploiting the ~72 nt periodicity.

Reads the test predictions saved by 04_train.py.  No retraining.

Usage:
    python 07_error_concentration.py --tag final
    python 07_error_concentration.py --tag final --widths 3 5 10 15
"""

import argparse
from pathlib import Path

import numpy as np

PE_ORDER  = ["learned", "sinusoidal", "rope", "alibi"]
PE_LABELS = {"learned": "Learned Abs.", "sinusoidal": "Sinusoidal",
             "rope": "RoPE",            "alibi": "ALiBi"}
SEEDS = [0, 1, 2, 3, 4, 5]


def split_runs(pos: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous segments; a break is any non-unit step in absolute position."""
    if len(pos) == 0:
        return []
    breaks = np.nonzero(np.diff(pos) != 1)[0] + 1
    edges  = np.concatenate([[0], breaks, [len(pos)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1]


def edge_distance(labels: np.ndarray) -> np.ndarray:
    """
    Unsigned nt distance from each token to the nearest label transition.
    Two-pass scan: O(n), no distance matrix.
    """
    n = len(labels)
    change = np.nonzero(np.diff(labels) != 0)[0]      # boundary sits between i and i+1
    if len(change) == 0:
        return np.full(n, n, dtype=np.float64)

    bpos = change + 0.5
    idx  = np.arange(n, dtype=np.float64)
    j    = np.searchsorted(bpos, idx)                  # first boundary at or after idx
    left  = np.where(j > 0,        idx - bpos[np.clip(j - 1, 0, len(bpos) - 1)], np.inf)
    right = np.where(j < len(bpos), bpos[np.clip(j, 0, len(bpos) - 1)] - idx,   np.inf)
    return np.minimum(left, right)


def collect(pe: str, tag: str):
    """Returns (correct, dist) concatenated over every seed, or None."""
    root = Path(f"outputs/checkpoints_{tag}")
    C, D = [], []
    for s in SEEDS:
        f = root / pe / f"seed{s}" / "test_preds.npz"
        if not f.exists():
            continue
        d = np.load(f)
        preds, labels, pos = d["preds"], d["labels"], d["pos"]
        for a, b in split_runs(pos):
            L = labels[a:b]
            C.append((preds[a:b] == L).astype(np.float64))
            D.append(edge_distance(L))
    if not C:
        return None
    return np.concatenate(C), np.concatenate(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="final")
    ap.add_argument("--widths", nargs="+", type=int, default=[2, 3, 5, 8])
    ap.add_argument("--latex", action="store_true",
                    help="Also write a LaTeX table to outputs/figures_<tag>/")
    args = ap.parse_args()

    data = {pe: collect(pe, args.tag) for pe in PE_ORDER}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        print(f"[ERROR] No test_preds under outputs/checkpoints_{args.tag}/")
        return

    print(f"\n══ Error concentration at block boundaries ({args.tag}) ══\n")
    print("  enrich = share of errors in zone / share of tokens in zone")
    print("           1.0 means errors are spread uniformly along the array")
    print("  max    = largest enrichment attainable at this zone width")
    print("           (reached only if every error lies inside the zone).")
    print("           Blocks are ~36 nt, so wide zones cover most tokens and")
    print("           cap the attainable enrichment -- read enrich against max.\n")

    rows = {}
    for N in args.widths:
        print(f"  ── boundary zone = +/-{N} nt " + "─" * 42)
        print(f"  {'PE type':<14}{'tokens%':>9}{'errors%':>9}{'enrich':>8}"
              f"{'max':>7}{'acc_zone':>10}{'acc_int':>9}")
        for pe, (C, D) in data.items():
            zone     = D <= N
            err      = C == 0
            tok_pct  = 100 * zone.mean()
            err_pct  = 100 * (err & zone).sum() / max(err.sum(), 1)
            enrich   = (err_pct / tok_pct) if tok_pct > 0 else float("nan")
            acc_zone = C[zone].mean()  if zone.any()  else float("nan")
            acc_int  = C[~zone].mean() if (~zone).any() else float("nan")
            max_enr = (100.0 / tok_pct) if tok_pct > 0 else float("nan")
            print(f"  {PE_LABELS[pe]:<14}{tok_pct:>9.1f}{err_pct:>9.1f}"
                  f"{enrich:>8.2f}{max_enr:>7.2f}{acc_zone:>10.3f}{acc_int:>9.3f}")
            rows.setdefault(N, []).append(
                (pe, tok_pct, err_pct, enrich, acc_zone, acc_int))
        print()

    # Headline sentence built from the widest-consensus width
    N = 5 if 5 in args.widths else args.widths[len(args.widths) // 2]
    print("  ── summary " + "─" * 55)
    for pe, tok_pct, err_pct, enrich, acc_zone, acc_int in rows[N]:
        print(f"  {PE_LABELS[pe]:<14} {err_pct:.0f}% of errors fall in the "
              f"{tok_pct:.0f}% of tokens within +/-{N} nt of a boundary "
              f"({enrich:.1f}x enrichment);")
        print(f"  {'':<14} interior accuracy {acc_int:.3f} vs "
              f"{acc_zone:.3f} in the boundary zone.")

    if args.latex:
        out = Path(f"outputs/figures_{args.tag}") / "error_concentration.tex"
        out.parent.mkdir(parents=True, exist_ok=True)
        L = [r"\begin{table}[t]", r"\centering",
             r"\caption{Errors concentrate at repeat/spacer boundaries. "
             rf"Boundary zone is $\pm{N}$ nt around each block transition. "
             r"Enrichment is the share of errors in the zone divided by the "
             r"share of tokens in it; $1.0$ would mean errors are spread "
             r"uniformly. High interior accuracy with strong boundary "
             r"enrichment indicates composition matching rather than use of "
             r"the array's periodicity.}",
             r"\label{tab:error-concentration}",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Positional encoding & Tokens in zone & Errors in zone & "
             r"Enrichment & Acc. (zone) & Acc. (interior) \\", r"\midrule"]
        for pe, tok_pct, err_pct, enrich, acc_zone, acc_int in rows[N]:
            L.append(f"{PE_LABELS[pe]} & {tok_pct:.1f}\\% & {err_pct:.1f}\\% & "
                     f"{enrich:.2f}$\\times$ & {acc_zone:.3f} & {acc_int:.3f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        out.write_text("\n".join(L))
        print(f"\n  saved → {out}")


if __name__ == "__main__":
    main()

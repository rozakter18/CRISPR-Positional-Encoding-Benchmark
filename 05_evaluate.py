"""
05_evaluate.py
────────────────────────────────────────────────────────────────────────────────
Loads tagged test results and produces:

  1. Console summary — mean ± std for each PE type, with trivial baselines
  2. outputs/figures_<tag>/results.png        — bar chart with baseline lines
  3. outputs/figures_<tag>/per_position.png   — per-position accuracy over one period
  4. outputs/figures_<tag>/results_table.tex  — LaTeX table for the report
  5. Wilcoxon signed-rank tests across seeds

Trivial baselines are computed from the actual test-set label distribution and
drawn as reference lines, because a token-level F1 near 0.6 on a ~43% positive
task can be achieved by predicting the positive class everywhere.  Without
these lines a reader cannot tell a weak model from a degenerate one.

Usage:
    python 05_evaluate.py --tag v2
    python 05_evaluate.py --tag v2 --period 72
    python 05_evaluate.py --tag v2 --compare v1     # side-by-side ablation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from common import compute_per_position_accuracy

DATA_PATH = Path("outputs/datasets/dataset.npz")

PE_ORDER  = ["learned", "sinusoidal", "rope", "alibi"]
PE_LABELS = {
    "learned":    "Learned Abs.",
    "sinusoidal": "Sinusoidal",
    "rope":       "RoPE",
    "alibi":      "ALiBi",
}
PE_COLORS = {
    "learned":    "#6C757D",
    "sinusoidal": "#0D6EFD",
    "rope":       "#198754",
    "alibi":      "#E0A800",
}
SEEDS = [0, 1, 2, 3, 4, 5]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_metrics(tag: str) -> dict[str, list[dict]]:
    path = Path(f"outputs/metrics_{tag}.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python 04_train.py --tag {tag}"
        )
    results = json.loads(path.read_text())
    grouped = {}
    for r in results:
        grouped.setdefault(r["pe_type"], []).append(r)
    return grouped


def compute_baselines() -> dict:
    """
    Trivial baselines from the real test-set label distribution.

    all-repeat : predict class 1 everywhere  -> recall 1, precision = prevalence
    all-spacer : predict class 0 everywhere  -> F1 0, accuracy = 1 - prevalence
    random     : predict 1 with prob = prevalence
    """
    if not DATA_PATH.exists():
        return {}
    lbls = np.load(DATA_PATH)["test_lbls"].flatten()
    lbls = lbls[lbls != -100]
    prev = float(lbls.mean())

    return {
        "prevalence": prev,
        "all_repeat": {"f1": 2 * prev / (1 + prev), "acc": prev},
        "all_spacer": {"f1": 0.0,                   "acc": 1 - prev},
        "random":     {"f1": prev,                  "acc": prev**2 + (1-prev)**2},
    }


def summarize(grouped: dict) -> dict:
    out = {}
    for pe, runs in grouped.items():
        out[pe] = {
            m: (np.mean([r[f"test_{m}"] for r in runs]),
                np.std ([r[f"test_{m}"] for r in runs]))
            for m in ("f1", "accuracy", "precision", "recall")
        }
        out[pe]["n_runs"]   = len(runs)
        out[pe]["n_params"] = runs[0].get("n_params", 0)
    return out


def pairwise_wilcoxon(grouped: dict) -> list[tuple]:
    """All pairwise Wilcoxon signed-rank tests on test F1 across seeds."""
    rows = []
    pes  = [p for p in PE_ORDER if p in grouped]
    for i, a in enumerate(pes):
        for b in pes[i+1:]:
            fa = np.array([r["test_f1"] for r in grouped[a]])
            fb = np.array([r["test_f1"] for r in grouped[b]])
            n  = min(len(fa), len(fb))
            if n < 2 or np.allclose(fa[:n], fb[:n]):
                rows.append((a, b, float("nan"), n))
                continue
            p = wilcoxon(fa[:n], fb[:n]).pvalue
            rows.append((a, b, float(p), n))
    return rows


def robust_metrics(pe: str, tag: str) -> dict | None:
    """
    Balanced accuracy and Matthews correlation coefficient, per seed, from the
    saved test predictions.

    Repeat-class F1 is a poor headline metric on this task: predicting the
    positive class everywhere scores 2p/(1+p), which at ~50% prevalence beats
    a genuinely-trained model.  MCC is 0 for any constant predictor and
    balanced accuracy is 0.5, so both separate real learning from degeneracy.
    """
    root = Path(f"outputs/checkpoints_{tag}")
    bacc, mcc = [], []
    for s in SEEDS:
        f = root / pe / f"seed{s}" / "test_preds.npz"
        if not f.exists():
            continue
        d = np.load(f)
        y, p = d["labels"], d["preds"]
        tp = np.sum((p == 1) & (y == 1)); tn = np.sum((p == 0) & (y == 0))
        fp = np.sum((p == 1) & (y == 0)); fn = np.sum((p == 0) & (y == 1))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        bacc.append((tpr + tnr) / 2)
        den = np.sqrt(float(tp+fp) * float(tp+fn) * float(tn+fp) * float(tn+fn))
        mcc.append(float((tp*tn - fp*fn) / den) if den > 0 else 0.0)
    if not bacc:
        return None
    return {
        "bacc": (float(np.mean(bacc)), float(np.std(bacc))),
        "mcc":  (float(np.mean(mcc)),  float(np.std(mcc))),
        "n":    len(bacc),
    }


def load_test_preds(pe: str, tag: str):
    root = Path(f"outputs/checkpoints_{tag}")
    P, L, X = [], [], []
    for s in SEEDS:
        f = root / pe / f"seed{s}" / "test_preds.npz"
        if not f.exists():
            continue
        d = np.load(f)
        P.append(d["preds"]); L.append(d["labels"]); X.append(d["pos"])
    if not P:
        return None, None, None
    return np.concatenate(P), np.concatenate(L), np.concatenate(X)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_results(summary: dict, base: dict, tag: str, out: Path, rob: dict | None = None):
    pes = [p for p in PE_ORDER if p in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    panels = [("f1", "Repeat-class F1"), ("accuracy", "Token accuracy")]
    if rob:
        panels[1] = ("mcc", "Matthews correlation (MCC)")

    for ax, (metric, title) in zip(axes, panels):
        if metric == "mcc":
            means = [rob[p]["mcc"][0] if p in rob else 0.0 for p in pes]
            stds  = [rob[p]["mcc"][1] if p in rob else 0.0 for p in pes]
        else:
            means = [summary[p][metric][0] for p in pes]
            stds  = [summary[p][metric][1] for p in pes]
        x = np.arange(len(pes))

        ax.bar(x, means, yerr=stds, width=0.56, capsize=5,
               color=[PE_COLORS[p] for p in pes],
               edgecolor="black", linewidth=0.6, alpha=0.9, zorder=3)

        if metric == "mcc":
            ax.axhline(0.0, ls="--", lw=1.3, color="#C1121F", zorder=4,
                       label="any constant predictor (0.000)")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        elif base:
            key = "f1" if metric == "f1" else "acc"
            ax.axhline(base["all_repeat"][key], ls="--", lw=1.3, color="#C1121F",
                       zorder=4, label=f"all-repeat ({base['all_repeat'][key]:.3f})")
            ax.axhline(base["all_spacer"][key], ls=":", lw=1.3, color="#495057",
                       zorder=4, label=f"all-spacer ({base['all_spacer'][key]:.3f})")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([PE_LABELS[p] for p in pes], fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f"{title} (↑ better)", fontsize=10)
        ax.set_ylim(min(0, min(means) - 0.05) if metric == "mcc" else 0, 1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, ls="--", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "CRISPR repeat localization — positional encoding benchmark "
        f"(S. islandicus, {tag})",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


def plot_per_position(period: int, tag: str, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    any_data = False

    for pe in PE_ORDER:
        preds, labels, pos = load_test_preds(pe, tag)
        if preds is None:
            continue
        any_data = True
        acc = compute_per_position_accuracy(
            preds.tolist(), labels.tolist(), pos.tolist(), period=period
        )
        ax.plot(np.arange(period), acc, lw=2.0, alpha=0.88,
                color=PE_COLORS[pe], label=PE_LABELS[pe])

    if not any_data:
        plt.close()
        print("  [skip] no test_preds found for per-position plot")
        return

    ax.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel("Position within one repeat–spacer period (nt)", fontsize=11)
    ax.set_ylabel("Token accuracy", fontsize=11)
    ax.set_title(
        f"Per-position accuracy across one ~{period} bp repeat–spacer period ({tag})",
        fontsize=11,
    )
    ax.set_xlim(0, period - 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


def write_latex(summary: dict, base: dict, rob: dict, tag: str, out: Path):
    pes = [p for p in PE_ORDER if p in summary]
    L = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{CRISPR repeat-vs-spacer localization in \textit{S. islandicus}. "
        r"Token-level test metrics, mean $\pm$ std over 6 seeds. MCC and balanced "
        r"accuracy are reported because repeat-class F1 rewards degenerate "
        r"constant predictors at this label prevalence: predicting the positive "
        r"class everywhere attains F1 "
        + (f"${base['all_repeat']['f1']:.3f}$" if base else "$2p/(1{+}p)$")
        + r", while any constant predictor attains MCC $0$ and balanced accuracy $0.5$.}",
        r"\label{tab:pe-results}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Positional encoding & MCC & Balanced acc. & F1 (repeat) & Precision & Recall \\",
        r"\midrule",
    ]
    for p in pes:
        s = summary[p]
        r = rob.get(p)
        mcc  = f"${r['mcc'][0]:.3f} \\pm {r['mcc'][1]:.3f}$"   if r else "--"
        bacc = f"${r['bacc'][0]:.3f} \\pm {r['bacc'][1]:.3f}$" if r else "--"
        L.append(
            f"{PE_LABELS[p]} & {mcc} & {bacc} & "
            f"${s['f1'][0]:.3f} \\pm {s['f1'][1]:.3f}$ & "
            f"${s['precision'][0]:.3f} \\pm {s['precision'][1]:.3f}$ & "
            f"${s['recall'][0]:.3f} \\pm {s['recall'][1]:.3f}$ \\\\"
        )
    if base:
        L += [r"\midrule",
              f"Predict all repeat & $0.000$ & $0.500$ & "
              f"${base['all_repeat']['f1']:.3f}$ & "
              f"${base['prevalence']:.3f}$ & $1.000$ \\\\",
              r"Predict all spacer & $0.000$ & $0.500$ & $0.000$ & -- & $0.000$ \\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(L))
    print(f"  saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",     type=str, default="v2")
    ap.add_argument("--period",  type=int, default=72)
    ap.add_argument("--compare", type=str, default=None,
                    help="Second tag to print alongside (e.g. v1) as an ablation.")
    args = ap.parse_args()

    grouped = load_metrics(args.tag)
    summary = summarize(grouped)
    base    = compute_baselines()

    fig_dir = Path(f"outputs/figures_{args.tag}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    print(f"\n══ Results ({args.tag}) ═══════════════════════════════════════════════")
    if base:
        print(f"Test-set repeat prevalence: {base['prevalence']:.3f}\n")
    hdr = f"{'PE type':<14}{'F1':>16}{'Accuracy':>16}{'Precision':>16}{'Recall':>16}{'n':>4}"
    print(hdr); print("─" * len(hdr))
    for p in PE_ORDER:
        if p not in summary:
            continue
        s = summary[p]
        print(f"{PE_LABELS[p]:<14}"
              f"{s['f1'][0]:>9.4f}±{s['f1'][1]:.4f}"
              f"{s['accuracy'][0]:>9.4f}±{s['accuracy'][1]:.4f}"
              f"{s['precision'][0]:>9.4f}±{s['precision'][1]:.4f}"
              f"{s['recall'][0]:>9.4f}±{s['recall'][1]:.4f}"
              f"{s['n_runs']:>4}")

    if base:
        print("─" * len(hdr))
        print(f"{'all-repeat':<14}{base['all_repeat']['f1']:>9.4f}       "
              f"{base['all_repeat']['acc']:>9.4f}        (trivial baseline)")
        print(f"{'all-spacer':<14}{base['all_spacer']['f1']:>9.4f}       "
              f"{base['all_spacer']['acc']:>9.4f}        (trivial baseline)")
        print(f"{'random':<14}{base['random']['f1']:>9.4f}       "
              f"{base['random']['acc']:>9.4f}        (trivial baseline)")

    # ── Prevalence-robust metrics ─────────────────────────────────────────────
    print("\n── Prevalence-robust metrics (from saved test predictions) ──")
    print("   Any constant predictor scores MCC 0.000 and balanced acc 0.500.")
    rob = {p: robust_metrics(p, args.tag) for p in PE_ORDER}
    rob = {k: v for k, v in rob.items() if v is not None}
    if rob:
        print(f"\n  {'PE type':<14}{'Balanced acc':>18}{'MCC':>18}{'verdict':>26}")
        print("  " + "─" * 74)
        for p in PE_ORDER:
            r = rob.get(p)
            if r is None:
                continue
            m = r["mcc"][0]
            verdict = ("learned the task"        if m > 0.20 else
                       "weak but above chance"   if m > 0.05 else
                       "NEAR-DEGENERATE")
            print(f"  {PE_LABELS[p]:<14}"
                  f"{r['bacc'][0]:>11.4f}±{r['bacc'][1]:.4f}"
                  f"{r['mcc'][0]:>11.4f}±{r['mcc'][1]:.4f}"
                  f"{verdict:>26}")
    else:
        print("  [skip] no test_preds found — rerun 04_train.py to generate them")

    if base:
        print("\n── F1 vs trivial baselines (context only; prefer MCC above) ──")
        for p in PE_ORDER:
            if p not in summary:
                continue
            f1_ok  = summary[p]["f1"][0]       > base["all_repeat"]["f1"]
            acc_ok = summary[p]["accuracy"][0] > base["all_spacer"]["acc"]
            print(f"  {PE_LABELS[p]:<14} F1>all-repeat: {str(f1_ok):<6} "
                  f"acc>all-spacer: {str(acc_ok)}")

    # ── Significance ──────────────────────────────────────────────────────────
    print("\n── Pairwise Wilcoxon signed-rank on test F1 ──")
    for a, b, p, n in pairwise_wilcoxon(grouped):
        if np.isnan(p):
            print(f"  {PE_LABELS[a]:<14} vs {PE_LABELS[b]:<14} n/a")
            continue
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"  {PE_LABELS[a]:<14} vs {PE_LABELS[b]:<14} p={p:.4f}  {star}  (n={n})")
    print("  Note: with n=6 seeds the minimum attainable two-sided p is 0.03125,")
    print("        so no pairwise test can reach p<0.05 at this seed count.")

    # ── Optional ablation comparison ──────────────────────────────────────────
    if args.compare:
        try:
            other = summarize(load_metrics(args.compare))
            print(f"\n── Ablation: {args.tag} vs {args.compare} (test F1) ──")
            print(f"  {'PE type':<14}{args.tag:>14}{args.compare:>14}{'Δ':>10}")
            for p in PE_ORDER:
                if p in summary and p in other:
                    a, b = summary[p]["f1"][0], other[p]["f1"][0]
                    print(f"  {PE_LABELS[p]:<14}{a:>14.4f}{b:>14.4f}{a-b:>+10.4f}")
        except FileNotFoundError as e:
            print(f"\n[skip ablation] {e}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures…")
    plot_results(summary, base, args.tag, fig_dir / "results.png", rob)
    plot_per_position(args.period, args.tag, fig_dir / "per_position.png")
    write_latex(summary, base, rob, args.tag, fig_dir / "results_table.tex")

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()



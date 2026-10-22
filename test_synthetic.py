"""
test_synthetic.py — End-to-end pipeline test using synthetic CRISPR arrays.

Generates realistic-looking S. islandicus-style CRISPR data (strict
repeat-spacer alternation), runs 2 PE types × 2 seeds (fast), and checks
that the training and evaluation scripts produce sensible output.

Usage: python test_synthetic.py
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ── Synthetic data ────────────────────────────────────────────────────────────

REPEAT_CONSENSUS = "GTTTTAGAGCTATGCTGTTTTGAATGGTCCCAAAAC"  # 36 bp
BASES = "ATGC"

def mutate(seq: str, rate: float = 0.05) -> str:
    out = []
    for c in seq:
        if random.random() < rate:
            out.append(random.choice(BASES))
        else:
            out.append(c)
    return "".join(out)

def random_spacer(length: int = 36) -> str:
    return "".join(random.choices(BASES, k=length))

def make_array(n_units: int = 20, mut_rate: float = 0.05) -> tuple[str, np.ndarray]:
    """Returns (sequence_str, label_array) for one CRISPR array."""
    seq, labels = [], []
    for _ in range(n_units):
        repeat  = mutate(REPEAT_CONSENSUS, mut_rate)
        spacer  = random_spacer(random.randint(34, 38))
        seq.extend(list(repeat))
        labels.extend([1] * len(repeat))
        seq.extend(list(spacer))
        labels.extend([0] * len(spacer))
    return "".join(seq), np.array(labels, dtype=np.int64)


def build_synthetic_dataset(
    n_arrays: int = 80,
    window: int = 512,
    stride: int = 128,
    seed: int = 42,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    arrays, label_arrays = [], []
    for _ in range(n_arrays):
        n_units = random.randint(15, 30)
        seq, lbl = make_array(n_units)
        arrays.append(seq)
        label_arrays.append(lbl)

    # Cluster-level split (simple random here — no clustering needed for synthetic)
    idx = list(range(n_arrays))
    random.shuffle(idx)
    n_tr, n_va = int(0.7 * n_arrays), int(0.15 * n_arrays)
    splits = {
        "train": idx[:n_tr],
        "val":   idx[n_tr:n_tr + n_va],
        "test":  idx[n_tr + n_va:],
    }

    def slide(indices):
        seqs, lbls, pos = [], [], []
        for i in indices:
            s, l = arrays[i], label_arrays[i]
            n = len(s)
            from common import tokenize_seq
            start = 0
            while start + window <= n:
                seqs.append(tokenize_seq(s[start:start + window]))
                lbls.append(l[start:start + window])
                pos.append(np.arange(start, start + window))
                start += stride
        return np.stack(seqs), np.stack(lbls), np.stack(pos)

    result = {}
    for split, idxs in splits.items():
        result[f"{split}_seqs"], result[f"{split}_lbls"], result[f"{split}_pos"] = slide(idxs)

    return result


# ── Quick training test (2 PE types × 2 seeds) ────────────────────────────────

def test_pipeline():
    print("── Generating synthetic dataset ─────────────────────────────────────")
    data = build_synthetic_dataset(n_arrays=80)

    # Keep smoke-test artifacts separate from the real benchmark outputs.
    tag = "synthetic_smoke"
    ckpt_root = Path(f"outputs/checkpoints_{tag}")
    fig_dir = Path(f"outputs/figures_{tag}")
    out_path = Path(f"outputs/datasets/dataset_{tag}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data, window_len=np.array(512))
    for split in ("train", "val", "test"):
        print(f"  {split}: {data[f'{split}_seqs'].shape[0]} windows")

    # Label balance
    all_lbls = np.concatenate([data["train_lbls"].flatten(),
                                data["val_lbls"].flatten(),
                                data["test_lbls"].flatten()])
    print(f"  Repeat: {100*all_lbls.mean():.1f}%  Spacer: {100*(1-all_lbls.mean()):.1f}%")

    print("\n── Training 2 PE types × 2 seeds (fast smoke-test) ──────────────────")

    # Inline minimal training (avoids subprocess; tests imports and model logic)
    import importlib
    train_mod = importlib.import_module("04_train")
    # Override to fast settings for the smoke test
    train_mod.MAX_EPOCHS = 12
    train_mod.PATIENCE   = 4
    train_mod.BATCH_SIZE = 16

    split_data = {
        "train": (data["train_seqs"], data["train_lbls"], data["train_pos"]),
        "val":   (data["val_seqs"],   data["val_lbls"],   data["val_pos"]),
        "test":  (data["test_seqs"],  data["test_lbls"],  data["test_pos"]),
    }

    all_results = []
    for pe_type in ("rope", "learned"):
        for seed in (0, 1):
            print(f"\n  PE={pe_type}  seed={seed}")
            result = train_mod.train_one_run(
                pe_type, seed, split_data, ckpt_root,
                scale_emb=True, min_epochs=4, max_epochs=12
            )
            print(f"    test F1={result['test_f1']:.4f}  acc={result['test_accuracy']:.4f}")
            all_results.append(result)

    metrics_path = Path(f"outputs/metrics_{tag}.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n-- Evaluation plots (minimal) --")
    eval_mod = importlib.import_module("05_evaluate")
    grouped    = {"learned": [], "sinusoidal": [], "rope": [], "alibi": []}
    for r in all_results:
        grouped[r["pe_type"]].append(r)
    # Remove empty PE types so evaluate doesn't crash
    grouped = {k: v for k, v in grouped.items() if v}

    summary    = eval_mod.summarize(grouped)
    print("\n  Results so far:")
    for pe, s in summary.items():
        print(f"    {pe:<14}  F1={s['f1'][0]:.4f} +/- {s['f1'][1]:.4f}  acc={s['accuracy'][0]:.4f} +/- {s['accuracy'][1]:.4f}")

    fig_dir.mkdir(parents=True, exist_ok=True)
    eval_mod.plot_per_position(72, tag, fig_dir / "per_position_test.png")

    print("\nSynthetic pipeline test passed.")
    print(f"  Synthetic artifacts kept separate under tag: {tag}")


if __name__ == "__main__":
    test_pipeline()


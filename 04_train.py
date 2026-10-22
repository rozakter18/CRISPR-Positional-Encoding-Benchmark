"""
04_train.py
────────────────────────────────────────────────────────────────────────────────
Trains all four CRISPRTransformer variants (learned, sinusoidal, rope, alibi)
over 5 seeds each.  Saves per-run metrics and the best checkpoint.

Usage:
    python 04_train.py --tag v2                      # full run, 4 x 5 seeds
    python 04_train.py --tag v2 --pe sinusoidal      # triage one PE first
    python 04_train.py --tag v1_ablation --no-scale  # reproduce unscaled v1

Outputs (tagged so earlier runs are never overwritten):
    outputs/checkpoints_<tag>/{pe}/seed{s}/best.pt
    outputs/checkpoints_<tag>/{pe}/seed{s}/history.npz
    outputs/checkpoints_<tag>/{pe}/seed{s}/test_preds.npz
    outputs/metrics_<tag>.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import CRISPRDataset, compute_metrics, set_seed

# Module name starts with a digit — must use importlib
import importlib
_mod = importlib.import_module("03_models")
CRISPRTransformer = _mod.CRISPRTransformer
PE_TYPES          = _mod.PE_TYPES

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("outputs/datasets/dataset.npz")
BATCH_SIZE   = 32
MAX_EPOCHS   = 80
PATIENCE     = 10        # early stopping on val repeat-F1
LR           = 1e-3      # identical across all PE types (no confound)
WEIGHT_DECAY = 1e-4
SEEDS        = [0, 1, 2, 3, 4]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(path: Path) -> dict:
    d = np.load(path)
    return {
        "train": (d["train_seqs"], d["train_lbls"], d["train_pos"]),
        "val":   (d["val_seqs"],   d["val_lbls"],   d["val_pos"]),
        "test":  (d["test_seqs"],  d["test_lbls"],  d["test_pos"]),
    }


def make_loaders(split_data: dict, batch_size: int):
    loaders = {}
    for split, (seqs, lbls, _pos) in split_data.items():
        ds = CRISPRDataset(list(seqs), list(lbls))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            collate_fn=CRISPRDataset.collate, num_workers=0,
        )
    return loaders


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    """One epoch.  optimizer=None -> eval mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for seqs, lbls, pad_mask in loader:
            seqs, lbls, pad_mask = seqs.to(device), lbls.to(device), pad_mask.to(device)
            logits = model(seqs, pad_mask)

            logits_flat = logits.view(-1, 2)
            lbls_flat   = lbls.view(-1)
            loss        = criterion(logits_flat, lbls_flat)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

            preds = logits_flat.argmax(-1).cpu().numpy()
            labs  = lbls_flat.cpu().numpy()
            valid = labs != -100
            all_preds.extend(preds[valid].tolist())
            all_labels.extend(labs[valid].tolist())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def train_one_run(pe_type: str, seed: int, split_data: dict,
                  ckpt_root: Path, scale_emb: bool = True,
                  min_epochs: int = 25, max_epochs: int = MAX_EPOCHS) -> dict:
    set_seed(seed)

    ckpt_dir = ckpt_root / pe_type / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loaders   = make_loaders(split_data, BATCH_SIZE)
    model     = CRISPRTransformer(pe_type=pe_type, scale_emb=scale_emb).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1, patience_ctr = -1.0, 0
    history = {"train_loss": [], "val_f1": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        tr_m  = run_epoch(model, loaders["train"], criterion, optimizer)
        val_m = run_epoch(model, loaders["val"],   criterion)

        history["train_loss"].append(tr_m["loss"])
        history["val_f1"].append(val_m["f1"])
        history["val_loss"].append(val_m["loss"])

        if val_m["f1"] > best_val_f1:
            best_val_f1, patience_ctr = val_m["f1"], 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d}  tr_loss={tr_m['loss']:.4f}  "
                  f"val_f1={val_m['f1']:.4f}  val_acc={val_m['accuracy']:.4f}  "
                  f"patience={patience_ctr}/{PATIENCE}")

        # Early stopping is suppressed until min_epochs so that every PE type
        # gets an equal minimum training budget.  Without this floor, a config
        # that plateaus early (sinusoidal) is terminated during its plateau
        # while a faster starter (RoPE) trains 3-5x longer — an unequal-budget
        # confound, not a property of the encoding.
        if epoch >= min_epochs and patience_ctr >= PATIENCE:
            print(f"    Early stop at epoch {epoch}.")
            break

    np.savez(ckpt_dir / "history.npz",
             train_loss=np.array(history["train_loss"]),
             val_f1=np.array(history["val_f1"]),
             val_loss=np.array(history["val_loss"]))

    # Test with the best checkpoint
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=DEVICE))
    test_m = run_epoch(model, loaders["test"], criterion)

    # Save raw predictions + positions for the per-position accuracy plot
    model.eval()
    te_seqs, te_lbls, te_pos = split_data["test"]
    te_loader = DataLoader(CRISPRDataset(list(te_seqs), list(te_lbls)),
                           batch_size=BATCH_SIZE, collate_fn=CRISPRDataset.collate)

    all_preds, all_labels, all_pos = [], [], []
    idx = 0
    with torch.no_grad():
        for seqs, lbls, pad_mask in te_loader:
            logits = model(seqs.to(DEVICE), pad_mask.to(DEVICE))
            preds  = logits.argmax(-1).cpu().numpy()
            labs   = lbls.numpy()
            for b in range(seqs.size(0)):
                valid = labs[b] != -100
                all_preds.extend(preds[b][valid].tolist())
                all_labels.extend(labs[b][valid].tolist())
                pos_b = te_pos[idx + b]
                all_pos.extend(pos_b[valid[:pos_b.shape[0]]].tolist())
            idx += seqs.size(0)

    np.savez(ckpt_dir / "test_preds.npz",
             preds=np.array(all_preds),
             labels=np.array(all_labels),
             pos=np.array(all_pos))

    return {
        "pe_type":   pe_type,
        "seed":      seed,
        "n_params":  model.n_params,
        "scale_emb":  scale_emb,
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        **{f"test_{k}": v for k, v in test_m.items()},
        "best_val_f1": best_val_f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="v2",
                    help="Run label; keeps outputs separate from earlier runs.")
    ap.add_argument("--pe", nargs="+", default=list(PE_TYPES), choices=list(PE_TYPES),
                    help="Which PE type(s) to train (default: all four).")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--no-scale", action="store_true",
                    help="Disable sqrt(d_model) embedding scaling (reproduces v1).")
    ap.add_argument("--redo", action="store_true",
                    help="Re-run the selected --pe types even if results for "
                         "this tag already exist (replaces them in place).")
    ap.add_argument("--min-epochs", type=int, default=25,
                    help="Suppress early stopping until this epoch so every PE "
                         "type gets an equal minimum budget (default: 25).")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                    help=f"Hard epoch cap (default: {MAX_EPOCHS}).")
    ap.add_argument("--fixed-budget", action="store_true",
                    help="Disable early stopping entirely: train every run for "
                         "exactly --max-epochs and keep the best-val checkpoint. "
                         "Equal budget by construction — removes the stopping "
                         "rule as a confound between PE types.")
    args = ap.parse_args()

    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATA_PATH}. Run 02_parse_and_build.py first.")
        return

    scale_emb   = not args.no_scale
    # --fixed-budget: floor == cap, so the patience test can never fire
    min_ep      = args.max_epochs if args.fixed_budget else args.min_epochs
    ckpt_root   = Path(f"outputs/checkpoints_{args.tag}")
    metrics_out = Path(f"outputs/metrics_{args.tag}.json")

    print(f"Device: {DEVICE}")
    mode = (f"fixed budget, {args.max_epochs} epochs, no early stop"
            if args.fixed_budget
            else f"early stop after epoch {min_ep}, patience {PATIENCE}, "
                 f"cap {args.max_epochs}")
    print(f"Tag: {args.tag}   embedding scaling: "
          f"{'ON' if scale_emb else 'OFF'}   {mode}")

    split_data = load_dataset(DATA_PATH)
    for split, (seqs, _l, _p) in split_data.items():
        print(f"  {split}: {seqs.shape[0]} windows")

    # Resume support: keep results from a previous partial run with this tag
    all_results = []
    if metrics_out.exists():
        all_results = json.loads(metrics_out.read_text())
        if args.redo:
            # Drop existing results for the PE types being re-run
            before = len(all_results)
            all_results = [r for r in all_results if r["pe_type"] not in args.pe]
            print(f"  --redo: discarding {before - len(all_results)} existing "
                  f"run(s) for {', '.join(args.pe)}.")
        done = {(r["pe_type"], r["seed"]) for r in all_results}
        if not args.redo:
            print(f"  Resuming — {len(done)} run(s) already complete for tag '{args.tag}'.")
    else:
        done = set()

    total = len(args.pe) * len(args.seeds)
    n = 0
    for pe_type in args.pe:
        for seed in args.seeds:
            n += 1
            if (pe_type, seed) in done:
                print(f"\n── [{n}/{total}]  PE={pe_type}  seed={seed}  — already done, skipping")
                continue
            print(f"\n── [{n}/{total}]  PE={pe_type}  seed={seed}  ──────────────────────")
            t0 = time.time()
            result = train_one_run(pe_type, seed, split_data, ckpt_root,
                                   scale_emb, min_ep, args.max_epochs)
            print(f"  → test F1={result['test_f1']:.4f}  "
                  f"acc={result['test_accuracy']:.4f}  ({time.time()-t0:.0f}s)")
            all_results.append(result)
            metrics_out.parent.mkdir(parents=True, exist_ok=True)
            metrics_out.write_text(json.dumps(all_results, indent=2))

    print(f"\n✓ Done.  Metrics → {metrics_out}")
    print(f"  Evaluate with:  python 05_evaluate.py --tag {args.tag}")


if __name__ == "__main__":
    main()
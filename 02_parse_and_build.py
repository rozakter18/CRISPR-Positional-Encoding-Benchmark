"""
02_parse_and_build.py
──────────────────────────────────────────────────────────────────────────────
Extracts CRISPR arrays from S. islandicus GenBank files and builds the
token-labelled dataset for the positional-encoding benchmark.

Detection and labelling are handled by crispr_finder.py, which uses vectorised
mismatch-tolerant self-comparison followed by iterative consensus refinement.
This tolerates both per-repeat sequence divergence (8-15%) and variable spacer
lengths (30-45 bp), neither of which exact k-mer matching or autocorrelation
can handle.

Mode A (annotated): run the finder inside each annotated repeat_region feature.
Mode B (de novo):   run the finder across the whole genome.
Entries below MIN_GENOME_LEN are skipped (plasmids, fragments, RNA genes).

Usage:
    python 02_parse_and_build.py [--window 512] [--stride 128]
"""

import argparse, random, sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

from common import tokenize_seq, set_seed
from crispr_finder import find_crispr_arrays, REPEAT_LEN, MIN_COPIES

GENOMES_DIR    = Path("data/genomes")
OUT_DIR        = Path("outputs/datasets")
MIN_GENOME_LEN = 100_000


# ── Per-genome processing ─────────────────────────────────────────────────────

def process_genome(record) -> tuple[list[dict], str]:
    """Returns (arrays, mode) with mode in {'annotated', 'denovo', 'skipped'}."""
    if len(record.seq) < MIN_GENOME_LEN:
        return [], "skipped"
    try:
        seq_str = str(record.seq)
    except Exception:
        return [], "skipped"   # metadata-only WGS master records

    # ── Mode A: search inside annotated repeat_region features ────────────────
    annotated = []
    for feat in record.features:
        if feat.type != "repeat_region":
            continue
        note = " ".join(
            feat.qualifiers.get("note", []) + feat.qualifiers.get("function", [])
        )
        if "CRISPR" not in note.upper():
            if feat.qualifiers.get("rpt_family", [""])[0].upper() != "CRISPR":
                continue
        try:
            region_seq = str(feat.extract(record.seq))
        except Exception:
            continue
        for a in find_crispr_arrays(region_seq):
            a["accession"] = record.id
            a["source"]    = "annotated"
            annotated.append(a)

    if annotated:
        return annotated, "annotated"

    # ── Mode B: whole-genome de novo scan ─────────────────────────────────────
    denovo = find_crispr_arrays(seq_str)
    for a in denovo:
        a["accession"] = record.id
        a["source"]    = "denovo"
    return denovo, "denovo"


# ── Clustering and splitting ──────────────────────────────────────────────────

def cluster_arrays(arrays: list[dict], k: int = 8) -> list[int]:
    """k-mer Jaccard single-linkage clustering (~90% identity proxy)."""
    n           = len(arrays)
    cluster_ids = list(range(n))
    sets = [set(a["seq"][i:i+k] for i in range(len(a["seq"]) - k + 1)) for a in arrays]
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_ids[i] == cluster_ids[j]:
                continue
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union > 0 and inter / union >= 0.8:
                old, new = cluster_ids[j], cluster_ids[i]
                cluster_ids = [new if c == old else c for c in cluster_ids]
    unique = sorted(set(cluster_ids))
    remap  = {v: i for i, v in enumerate(unique)}
    return [remap[c] for c in cluster_ids]


def cluster_split(cluster_ids, train_f=0.70, val_f=0.15, seed=42):
    rng      = random.Random(seed)
    clusters = sorted(set(cluster_ids))
    rng.shuffle(clusters)
    n_tr = max(1, int(len(clusters) * train_f))
    n_va = max(1, int(len(clusters) * val_f))
    tr, va, te = (set(clusters[:n_tr]),
                  set(clusters[n_tr:n_tr + n_va]),
                  set(clusters[n_tr + n_va:]))
    return (
        [i for i, c in enumerate(cluster_ids) if c in tr],
        [i for i, c in enumerate(cluster_ids) if c in va],
        [i for i, c in enumerate(cluster_ids) if c in te],
    )


# ── Windowing ─────────────────────────────────────────────────────────────────

def slide_windows(arrays, indices, win_len, stride):
    seqs, lbls, pos_idxs = [], [], []
    for idx in indices:
        a   = arrays[idx]
        seq = a["seq"]
        lab = np.array(a["labels"], dtype=np.int64)
        n   = len(seq)
        if n < win_len:
            seq = seq + "N" * (win_len - n)
            lab = np.concatenate([lab, np.zeros(win_len - n, dtype=np.int64)])
            n   = win_len
        start = 0
        while start + win_len <= n:
            seqs.append(tokenize_seq(seq[start:start + win_len]))
            lbls.append(lab[start:start + win_len])
            pos_idxs.append(np.arange(start, start + win_len, dtype=np.int64))
            start += stride
    return seqs, lbls, pos_idxs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gb_files = sorted(GENOMES_DIR.glob("*.gb"))
    if not gb_files:
        print(f"[ERROR] No .gb files in {GENOMES_DIR}. Run 01_fetch_genomes.py first.")
        sys.exit(1)

    print(f"Processing {len(gb_files)} GenBank file(s)  (skipping < {MIN_GENOME_LEN:,} bp)…\n")
    print(f"{'Accession':<22} {'Length':>10}  {'Mode':<10}  {'Arrays':>6}  "
          f"{'Repeats':>8}  {'RepeatPct':>10}")
    print("─" * 74)

    all_arrays, n_skipped, n_ann, n_dn = [], 0, 0, 0

    for gb in gb_files:
        for record in SeqIO.parse(gb, "genbank"):
            arrays, mode = process_genome(record)
            if mode == "skipped":
                n_skipped += 1
                continue

            n_rep     = sum(a["n_repeats"] for a in arrays)
            total_nt  = sum(len(a["seq"]) for a in arrays)
            rep_nt    = sum(int(np.array(a["labels"]).sum()) for a in arrays)
            rep_pct   = 100 * rep_nt / total_nt if total_nt else 0.0

            print(f"  {record.id:<20} {len(record.seq):>10,}  {mode:<10}  "
                  f"{len(arrays):>6}  {n_rep:>8}  {rep_pct:>9.1f}%")

            n_ann += len(arrays) if mode == "annotated" else 0
            n_dn  += len(arrays) if mode == "denovo"    else 0
            all_arrays.extend(arrays)

    print(f"\n{'─'*74}")
    print(f"  Skipped (< {MIN_GENOME_LEN:,} bp): {n_skipped}")
    print(f"  Mode A (annotated):  {n_ann} arrays")
    print(f"  Mode B (de novo):    {n_dn} arrays")
    print(f"  Total:               {len(all_arrays)} arrays")

    if not all_arrays:
        print("[ERROR] No CRISPR arrays extracted.")
        sys.exit(1)

    min_nt     = MIN_COPIES * REPEAT_LEN * 2
    all_arrays = [a for a in all_arrays if len(a["seq"]) >= min_nt]
    total_rep  = sum(a["n_repeats"] for a in all_arrays)
    print(f"  After length filter: {len(all_arrays)} arrays, {total_rep} repeat units\n")

    print("Clustering…", end=" ", flush=True)
    cluster_ids = cluster_arrays(all_arrays)
    print(f"{len(set(cluster_ids))} cluster(s)")

    tr_idx, va_idx, te_idx = cluster_split(cluster_ids, seed=args.seed)
    print(f"Cluster split: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)} arrays")

    tr_s, tr_l, tr_p = slide_windows(all_arrays, tr_idx, args.window, args.stride)
    va_s, va_l, va_p = slide_windows(all_arrays, va_idx, args.window, args.window)
    te_s, te_l, te_p = slide_windows(all_arrays, te_idx, args.window, args.window)
    print(f"Windows:       train={len(tr_s)}  val={len(va_s)}  test={len(te_s)}")

    if min(len(tr_s), len(va_s), len(te_s)) == 0:
        print("[ERROR] A split has zero windows. Reduce --window or check array yield.")
        sys.exit(1)

    out = OUT_DIR / "dataset.npz"
    np.savez_compressed(
        out,
        train_seqs=np.stack(tr_s), train_lbls=np.stack(tr_l), train_pos=np.stack(tr_p),
        val_seqs=np.stack(va_s),   val_lbls=np.stack(va_l),   val_pos=np.stack(va_p),
        test_seqs=np.stack(te_s),  test_lbls=np.stack(te_l),  test_pos=np.stack(te_p),
        window_len=np.array(args.window),
    )

    all_flat = np.concatenate(tr_l + va_l + te_l)
    rep_pct  = 100 * all_flat.mean()
    print(f"\nDataset → {out}")
    print(f"Label balance: {rep_pct:.1f}% repeat  /  {100-rep_pct:.1f}% spacer")
    print("✓ Balance looks good." if 30 <= rep_pct <= 65
          else "[WARN] Balance outside 30-65% — check the RepeatPct column above.")
    print("\nProceed to:  python 04_train.py")


if __name__ == "__main__":
    main()
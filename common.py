"""
common.py — Shared utilities for the CRISPR PE benchmark.
Tokenizer, dataset class, metrics, seeding.
"""

import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# ── Vocabulary ────────────────────────────────────────────────────────────────
# 6-token vocabulary: A T G C N PAD
# PAD index (5) used as padding_idx in nn.Embedding — never contributes to gradients

NT_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4, 'PAD': 5}
IDX_TO_NT = {v: k for k, v in NT_TO_IDX.items()}
VOCAB_SIZE = 6
PAD_IDX    = NT_TO_IDX['PAD']

# Label map: 0 = spacer (variable), 1 = repeat (conserved)
SPACER_LABEL = 0
REPEAT_LABEL = 1

# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize_seq(seq: str) -> np.ndarray:
    """Convert a DNA string to an integer array. Unknown chars → N."""
    return np.array(
        [NT_TO_IDX.get(c.upper(), NT_TO_IDX['N']) for c in seq],
        dtype=np.int64
    )

# ── Dataset ───────────────────────────────────────────────────────────────────

class CRISPRDataset(Dataset):
    """
    Holds windowed CRISPR array sequences and their per-nucleotide labels.

    Each item is a (seq_tensor, label_tensor) pair of identical length.
    Sequences are already fixed-length windows (no padding needed during training
    when batch_size is constant, but we keep a collate_fn for safety).
    """
    def __init__(self, sequences: list[np.ndarray], labels: list[np.ndarray]):
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels    = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.long),
        )

    @staticmethod
    def collate(batch):
        """Pad to the longest sequence in the batch."""
        seqs, lbls = zip(*batch)
        max_len = max(s.shape[0] for s in seqs)
        padded_seqs = torch.full((len(seqs), max_len), PAD_IDX,  dtype=torch.long)
        padded_lbls = torch.full((len(lbls), max_len), -100,     dtype=torch.long)  # -100 = ignored by CE loss
        for i, (s, l) in enumerate(zip(seqs, lbls)):
            padded_seqs[i, :s.shape[0]] = s
            padded_lbls[i, :l.shape[0]] = l
        pad_mask = (padded_seqs == PAD_IDX)
        return padded_seqs, padded_lbls, pad_mask

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_preds: list[int], all_labels: list[int]) -> dict:
    """
    Token-level metrics for the REPEAT class (class 1).
    Returns dict with accuracy, precision, recall, f1 for the repeat class.
    """
    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    # Overall accuracy (ignores -100 padding if caller already filtered)
    acc = (preds == labels).mean()

    # Per-class counts for the repeat label
    tp = ((preds == REPEAT_LABEL) & (labels == REPEAT_LABEL)).sum()
    fp = ((preds == REPEAT_LABEL) & (labels == SPACER_LABEL)).sum()
    fn = ((preds == SPACER_LABEL) & (labels == REPEAT_LABEL)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        'accuracy':  float(acc),
        'precision': float(precision),
        'recall':    float(recall),
        'f1':        float(f1),
    }

def compute_per_position_accuracy(
    all_preds:   list[int],
    all_labels:  list[int],
    all_pos_idx: list[int],
    period:      int = 72,
) -> np.ndarray:
    """
    Returns per-position accuracy across one averaged repeat-spacer period.
    all_pos_idx: absolute position of each token within its source array.
    period: repeat (~36 bp) + spacer (~36 bp) = ~72 bp.
    """
    preds   = np.array(all_preds)
    labels  = np.array(all_labels)
    pos_idx = np.array(all_pos_idx) % period

    correct_at = np.zeros(period, dtype=np.float32)
    count_at   = np.zeros(period, dtype=np.float32)
    for p in range(period):
        mask = (pos_idx == p)
        if mask.sum() > 0:
            correct_at[p] = (preds[mask] == labels[mask]).sum()
            count_at[p]   = mask.sum()

    return correct_at / (count_at + 1e-9)

# ── Seeding ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

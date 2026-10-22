"""
crispr_finder.py — mismatch-tolerant CRISPR array detection.

Exact k-mer anchoring fails on real arrays because individual repeats diverge
by 8-15%, so no single k-mer survives across all copies.  Autocorrelation fails
because spacer lengths vary (30-45 bp), so phase drifts along the array.

This module instead uses vectorised shifted self-comparison:

  Stage 1  For every candidate gap g in [MIN_GAP, MAX_GAP], compare the
           sequence to itself shifted by g and find all 36-bp windows whose
           Hamming distance is <= MAX_MM.  Fully vectorised with cumulative
           sums, so all gaps are screened in a few seconds even on a 4 Mb
           genome.  Positions flagged at any gap are candidate repeat starts.

  Stage 2  Group candidate positions into contiguous array regions.

  Stage 3  Within each region, iteratively refine a consensus repeat:
           seed -> greedy non-overlapping scan -> rebuild consensus -> rescan.
           Converges in 2-3 iterations and tolerates variable spacer lengths,
           because matching is by sequence rather than by fixed period.
"""

import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────
REPEAT_LEN   = 36    # canonical S. islandicus repeat length
MIN_GAP      = 50    # min distance between consecutive repeat starts
MAX_GAP      = 100   # max distance between consecutive repeat starts
MAX_MM       = 9     # max Hamming distance for a repeat match (25% of 36)
MIN_COPIES   = 3     # minimum repeat copies to call an array
REGION_GAP   = 150   # max distance between candidate hits in one array
N_REFINE     = 3     # consensus refinement iterations


def _to_arr(seq: str) -> np.ndarray:
    return np.frombuffer(seq.upper().encode(), dtype=np.uint8)


def find_candidate_positions(arr: np.ndarray,
                             rlen: int = REPEAT_LEN,
                             max_mm: int = MAX_MM) -> np.ndarray:
    """
    Stage 1: vectorised shifted self-comparison.
    Returns a boolean array; True at positions that start a repeat-like unit
    recurring at some gap in [MIN_GAP, MAX_GAP].
    """
    n    = len(arr)
    hits = np.zeros(n, dtype=bool)

    for g in range(MIN_GAP, MAX_GAP + 1):
        if n - g < rlen:
            break
        diff = (arr[:-g] != arr[g:]).astype(np.int32)
        cs   = np.concatenate([[0], np.cumsum(diff)])
        m    = len(diff)
        if m < rlen:
            continue
        # windowed mismatch count; index i = window starting at i
        wins = cs[rlen:m + 1] - cs[:m - rlen + 1]
        ok   = wins <= max_mm
        hits[:len(ok)] |= ok

    return hits


def group_regions(hits: np.ndarray,
                  region_gap: int = REGION_GAP,
                  min_span: int = MIN_GAP * MIN_COPIES) -> list[tuple[int, int]]:
    """
    Stage 2: group candidate positions into contiguous array regions.
    Returns a list of (start, end) spans.
    """
    idx = np.nonzero(hits)[0]
    if len(idx) == 0:
        return []

    regions = []
    lo = prev = idx[0]
    for p in idx[1:]:
        if p - prev > region_gap:
            if prev - lo >= min_span:
                regions.append((lo, prev))
            lo = p
        prev = p
    if prev - lo >= min_span:
        regions.append((lo, prev))

    return regions


def _scan_with_consensus(arr: np.ndarray,
                         consensus: np.ndarray,
                         max_mm: int = MAX_MM) -> list[int]:
    """Greedy non-overlapping scan for matches to `consensus`."""
    rlen = len(consensus)
    n    = len(arr)
    positions, i = [], 0
    while i <= n - rlen:
        mm = int((arr[i:i + rlen] != consensus).sum())
        if mm <= max_mm:
            positions.append(i)
            i += rlen
        else:
            i += 1
    return positions


def _build_consensus(arr: np.ndarray, positions: list[int],
                     rlen: int = REPEAT_LEN) -> np.ndarray | None:
    """Per-column majority vote across all repeat copies."""
    segs = [arr[p:p + rlen] for p in positions if p + rlen <= len(arr)]
    if len(segs) < MIN_COPIES:
        return None
    mat = np.vstack(segs)
    return np.array(
        [np.bincount(mat[:, c], minlength=256).argmax() for c in range(rlen)],
        dtype=np.uint8,
    )


def refine_region(arr: np.ndarray, lo: int, hi: int,
                  rlen: int = REPEAT_LEN) -> tuple[np.ndarray, list[int]] | None:
    """
    Stage 3: iterative consensus refinement within one region.
    Returns (labels, positions) relative to the region, or None.
    """
    pad    = MAX_GAP
    r_lo   = max(0, lo - pad)
    r_hi   = min(len(arr), hi + pad)
    region = arr[r_lo:r_hi]

    if len(region) < rlen * MIN_COPIES:
        return None

    # Seed from the first candidate hit inside the region, not the padded edge
    seed_off  = min(max(0, lo - r_lo), max(0, len(region) - rlen))
    consensus = region[seed_off:seed_off + rlen].copy()
    positions = []

    for _ in range(N_REFINE):
        positions = _scan_with_consensus(region, consensus)
        if len(positions) < MIN_COPIES:
            # Try a different seed offset before giving up
            found = False
            for off in range(0, max(1, len(region) - rlen), 3):
                cand = region[off:off + rlen].copy()
                p    = _scan_with_consensus(region, cand)
                if len(p) >= MIN_COPIES:
                    consensus, positions, found = cand, p, True
                    break
            if not found:
                return None
        new_consensus = _build_consensus(region, positions)
        if new_consensus is None:
            return None
        if np.array_equal(new_consensus, consensus):
            break
        consensus = new_consensus

    positions = _scan_with_consensus(region, consensus)
    if len(positions) < MIN_COPIES:
        return None

    # Verify the spacing looks CRISPR-like
    gaps = np.diff(positions)
    if len(gaps) and not (MIN_GAP - 10 <= np.median(gaps) <= MAX_GAP + 10):
        return None

    labels = np.zeros(len(region), dtype=np.int8)
    for p in positions:
        labels[p:min(p + rlen, len(region))] = 1

    # Trim to the repeat span plus one flanking spacer
    idx = np.nonzero(labels)[0]
    t_lo = max(0, idx[0] - 60)
    t_hi = min(len(region), idx[-1] + 1 + 60)

    return labels[t_lo:t_hi], [r_lo + t_lo, r_lo + t_hi]


def find_crispr_arrays(seq: str) -> list[dict]:
    """
    Full pipeline on a sequence (an annotated array or a whole genome).
    Returns a list of {seq, labels, locus, n_repeats} dicts.
    """
    arr = _to_arr(seq)
    if len(arr) < MIN_GAP * MIN_COPIES:
        return []

    hits    = find_candidate_positions(arr)
    regions = group_regions(hits)

    arrays = []
    for lo, hi in regions:
        result = refine_region(arr, lo, hi)
        if result is None:
            continue
        labels, (abs_lo, abs_hi) = result
        n_rep = int(
            np.diff(np.concatenate([[0], labels, [0]]).astype(int))
            .clip(min=0).sum()
        )
        if n_rep < MIN_COPIES:
            continue
        arrays.append({
            "seq":       seq[abs_lo:abs_hi].upper(),
            "labels":    labels.astype(np.int64),
            "locus":     f"{abs_lo}..{abs_hi}",
            "n_repeats": n_rep,
        })

    return arrays
# Repeat After Me

**Do rotary and relative positional encodings capture real periodic structure
better than absolute ones?**

Four transformer encoders, identical except for positional encoding, trained to
label every nucleotide of a CRISPR array as *repeat* or *spacer* in
*Sulfolobus islandicus* — an archaeon whose CRISPR loci alternate a conserved
~36 nt repeat with a variable spacer at a period of ~72 nt.

**Answer: no.** Sinusoidal (absolute) and RoPE (relative) tie on F1 and are nearly identical on MCC. The best
and worst encodings come from opposite families. Phase-locked analysis shows
why — no model uses the periodicity at all.

---

## Results

6 seeds, fixed 70-epoch budget, matched embedding scaling.

| Positional encoding | family | MCC | Balanced acc. | F1 (repeat) |
|---|---|---|---|---|
| Sinusoidal | absolute | 0.388 ± 0.029 | 0.697 | **0.667 ± 0.013** |
| RoPE | relative | **0.390 ± 0.036** | **0.697** | **0.667 ± 0.022** |
| Learned absolute | absolute | 0.364 ± 0.011 | 0.684 | 0.654 ± 0.007 |
| ALiBi | relative | 0.345 ± 0.030 | 0.672 | 0.616 ± 0.029 |
| *predict all repeat* | — | 0.000 | 0.500 | 0.588 |
| *predict all spacer* | — | 0.000 | 0.500 | 0.000 |

Sinusoidal vs RoPE: p = 0.563 (a genuine tie). Absolute vs relative as groups:
0.660 vs 0.641 mean F1, Mann–Whitney p = 0.31.

MCC is the headline metric: at 41.6% positive prevalence, predicting the
positive class everywhere scores F1 **0.588**, which beats a weakly-trained
model. Any constant predictor scores MCC 0.

### Why no encoding wins

All four localise block boundaries **8–10 nt late**, and that lag accounts for
**37–48% of all errors**. At the first nucleotide of a new block, accuracy is
0.17–0.30 — the models are still emitting the previous class. A model using the
~72 nt period would anticipate the transition. None does.

ALiBi differs in *error structure* despite similar aggregate accuracy: fastest
into spacers (floor 0.453, lag 6 nt), slowest and least accurate inside repeats
(plateau 0.796, lag 11 nt) — the predicted consequence of its linear distance
penalty suppressing the long-range comparisons that repeat recognition needs.

### Three confounds that fake a relative-encoding win

Each of these, left uncontrolled, independently manufactures a large apparent
advantage for relative encodings. All three point the same direction.

| # | Confound | Effect if uncontrolled | Control |
|---|---|---|---|
| 1 | Token embedding init at std 0.02 vs sinusoidal PE RMS 0.707 | position signal 36× token signal; sinusoidal pinned at chance (loss = ln 2) | scale embeddings by √d_model, init at d_model^−0.5 |
| 2 | Scaling token embedding but not the learned PE table | token signal 7.4× position; learned absolute drops 0.652 → 0.562 | scale the positional table identically |
| 3 | Early stopping with shared patience | absolute PEs plateau ~25–30 epochs before escaping; 9/10 absolute runs cut at the floor, 0/10 relative | fixed epoch budget, best-val checkpoint |

Uncontrolled, this pipeline reports sinusoidal MCC **0.117** vs RoPE **0.392**
— a decisive-looking win. Controlled: **0.388** vs **0.390**.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11, CPU-only is fine. No GPU required.

## Reproduce

```bash
python 01_fetch_genomes.py                                  # ~2 min, needs internet
python 02_parse_and_build.py                                # ~1 min
python 04_train.py --tag final --fixed-budget --max-epochs 70 --seeds 0 1 2 3 4 5
python 05_evaluate.py --tag final                           # aggregate metrics + figures
python 06_phase_analysis.py --tag final                     # phase-locked accuracy
python 07_error_concentration.py --tag final --latex        # boundary error enrichment
python 08_boundary_lag.py --tag final --latex               # boundary lag (main mechanism)
```

Training is the only slow step: ~28 min per run × 24 runs ≈ **11 hours on CPU**.
It checkpoints per run and resumes — rerunning the same `--tag` skips completed
runs, so an interruption costs only the run in flight.

To sanity-check the model code first:

```bash
python 03_models.py     # forward pass + token/position signal balance table
```

## Pipeline

| Script | Purpose |
|---|---|
| `01_fetch_genomes.py` | Fetch *S. islandicus* genomes from NCBI as GenBank |
| `02_parse_and_build.py` | Detect CRISPR arrays, label nucleotides, cluster, split, window |
| `crispr_finder.py` | Mismatch-tolerant CRISPR array detector with consensus refinement |
| `03_models.py` | Four PE variants on an identical encoder |
| `04_train.py` | Training loop, tagged outputs, resume support |
| `05_evaluate.py` | Aggregate metrics, trivial baselines, Wilcoxon, LaTeX table |
| `06_phase_analysis.py` | Phase-locked accuracy, drift-free binning |
| `07_error_concentration.py` | Error enrichment near block boundaries |
| `08_boundary_lag.py` | Boundary lag — the main mechanism result |
| `common.py` | Tokenizer, dataset, metrics, seeding |

Useful flags: `--tag` keeps result sets separate; `--redo` replaces existing
runs for the selected encodings; `--no-scale` reproduces the unscaled
configuration as an ablation; `--pe` and `--seeds` subset the runs.

## Dataset

Built from NCBI accessions by script — no manual curation.

- 59 GenBank records → 18 chromosomes ≥ 100 kb
- **50 CRISPR arrays, 1,773 repeat units**
- 48 clusters (8-mer Jaccard ≥ 0.8), split 70/15/15 at the **cluster** level
- 35 / 7 / 8 arrays → 678 / 26 / 40 windows of 512 nt
- 47.4% repeat overall; 41.6% in the test split

Cluster-level splitting matters: CRISPR arrays within a species are highly
similar, and a row-level split leaks near-duplicates across the boundary.

## Repo layout

- 01_fetch_genomes.py - fetch S. islandicus GenBank records from NCBI
- 02_parse_and_build.py - detect CRISPR arrays and build cluster-level dataset splits
- 03_models.py - learned, sinusoidal, RoPE, and ALiBi transformer variants
- 04_train.py - multi-seed training and checkpointing
- 05_evaluate.py - aggregate metrics, baselines, and statistical comparisons
- 06_phase_analysis.py - drift-free phase-locked accuracy analysis
- 07_error_concentration.py - error enrichment near repeat/spacer boundaries
- 08_boundary_lag.py - boundary-lag analysis
- common.py - shared tokenizer, dataset, metrics, and seeding utilities
- crispr_finder.py - mismatch-tolerant CRISPR array detector
- test_synthetic.py - synthetic end-to-end smoke test
- requirements.txt - Python dependencies
- data/genomes/ - downloaded NCBI records; regenerated locally and not tracked by Git
- outputs/metrics_final.json - final six-seed benchmark metrics
- outputs/figures_final/ - final figures and LaTeX result tables
- docs/crispr_pe_benchmark.pdf - full project report

Generated datasets, model checkpoints, intermediate experiments, synthetic smoke-test outputs, and downloaded genomes are excluded from version control and can be regenerated from the scripts.

## Limitations

- 8 test arrays (~20k tokens). Token metrics are stable; array-level diversity
  is low. Seed variance measures initialisation variance on a fixed split, not
  data variance.
- Validation F1 (0.72–0.79) exceeds test F1 (0.62–0.69) — checkpoint selection
  on 26 validation windows is noisy.
- Labels come from our detector, which stamps a fixed 36 nt repeat. True
  lengths vary by 1–2 nt, well below the 8–10 nt lag measured.
- d_model=64, 2 layers. Differences may emerge at scale. Sinusoidal was the
  least converged at the shared budget, so the tie with RoPE may not survive a
  longer run.

## Report

The full report is being updated to reflect the final six-seed evaluation and will be added here once reconciled with the released results.

## Author

Roza Akter — rozakterofficial18@gmail.com


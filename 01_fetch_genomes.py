"""
01_fetch_genomes.py
──────────────────────────────────────────────────────────────────────────────
Fetches S. islandicus complete genomes from NCBI (GenBank, not RefSeq-only).
Falls back to a hardcoded list of known Garrett-lab accessions if the query
yields too few results.

Run once before anything else.  Requires internet.
Usage:  python 01_fetch_genomes.py
"""

import os, time
from pathlib import Path
from io import StringIO

from Bio import Entrez, SeqIO

Entrez.email = "rozakterofficial18@gmail.com"
OUT_DIR      = Path("data/genomes")
FETCH_DELAY  = 0.4   # seconds between NCBI requests

# ── Primary NCBI query — GenBank + RefSeq, all complete S. islandicus ─────────
QUERY = (
    '"Sulfolobus islandicus"[Organism] '
    'AND ("complete genome"[Title] OR "complete sequence"[Title])'
)

# ── Known Garrett-lab accessions with explicit CRISPR annotations ──────────────
# These are GenBank (not RefSeq) and predate the RefSeq re-annotation that
# dropped the CRISPR repeat_region features.
KNOWN_ACCESSIONS = [
    "CP001390",   # REY15A     — Deng et al. 2009; 6 CRISPR loci
    "CP001401",   # M.16.27    — Reno et al. 2009
    "CP002556",   # L.S.2.15
    "CP002681",   # M.14.25
    "CP002682",   # M.16.4
    "CP003411",   # HVE10/4
    "CP003434",   # LAL14/1
    "CP006890",   # Y.N.15.51
    "CP006889",   # Y.G.57.14
    "CP006891",   # Y.N.15.26
    "CP007). 535", # placeholder — skip silently below if fetch fails
]
# Remove malformed placeholder
KNOWN_ACCESSIONS = [a for a in KNOWN_ACCESSIONS if ")" not in a]


def fetch_and_save(uid_or_acc: str, out_dir: Path) -> str | None:
    """Fetch a single record and save as <accession>.gb. Returns accession or None."""
    try:
        handle = Entrez.efetch(db="nucleotide", id=uid_or_acc,
                               rettype="gb", retmode="text")
        text = handle.read(); handle.close()
        record = next(SeqIO.parse(StringIO(text), "genbank"))
        acc = record.id
        path = out_dir / f"{acc}.gb"
        if path.exists():
            print(f"  {acc}  already on disk, skipping.")
            return acc
        path.write_text(text)
        return acc
    except Exception as e:
        print(f"  [WARN] Failed for {uid_or_acc}: {e}")
        return None


def crispr_feature_count(record) -> int:
    return sum(
        1 for f in record.features
        if f.type == "repeat_region"
        and any("CRISPR" in str(v) for v in f.qualifiers.values())
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: NCBI query ────────────────────────────────────────────────────
    print(f"[NCBI] Querying: {QUERY}")
    handle = Entrez.esearch(db="nucleotide", term=QUERY, retmax=50)
    result = Entrez.read(handle); handle.close()
    uids   = result["IdList"]
    print(f"[NCBI] Query returned {len(uids)} record(s).")

    downloaded = []
    for i, uid in enumerate(uids, 1):
        print(f"  [{i}/{len(uids)}] UID {uid} ...", end=" ", flush=True)
        acc = fetch_and_save(uid, OUT_DIR)
        if acc:
            print(f"→ {acc}")
            downloaded.append(acc)
        time.sleep(FETCH_DELAY)

    # ── Step 2: Fetch known accessions not already downloaded ─────────────────
    existing = {p.stem for p in OUT_DIR.glob("*.gb")}
    missing  = [a for a in KNOWN_ACCESSIONS if a not in existing]
    if missing:
        print(f"\nFetching {len(missing)} known accession(s) not covered by query…")
        for acc in missing:
            print(f"  {acc} ...", end=" ", flush=True)
            fetched = fetch_and_save(acc, OUT_DIR)
            if fetched:
                print(f"→ {fetched}")
                downloaded.append(fetched)
            time.sleep(FETCH_DELAY)

    # ── Preview ───────────────────────────────────────────────────────────────
    gb_files = sorted(OUT_DIR.glob("*.gb"))
    print(f"\n✓ {len(gb_files)} GenBank file(s) in {OUT_DIR}/")
    print("\n── CRISPR annotation preview ──────────────────────────────────────────")
    total_annotated = 0
    for gb in gb_files:
        record = next(SeqIO.parse(gb, "genbank"))
        n_crispr = crispr_feature_count(record)
        total_annotated += n_crispr
        flag = "✓" if n_crispr > 0 else "✗"
        print(f"  {flag} {record.id:<18} ({len(record.seq):>9,} bp)  "
              f"{n_crispr} CRISPR repeat_region feature(s)")

    if total_annotated == 0:
        print(
            "\n[NOTE] No CRISPR repeat_region features found in any genome.\n"
            "  02_parse_and_build.py will run a de novo array finder automatically.\n"
            "  This is expected for RefSeq-reannotated entries and is handled."
        )
    else:
        print(f"\n  {total_annotated} annotated CRISPR feature(s) total — "
              "02_parse_and_build.py will use these directly.")

    (OUT_DIR / "manifest.txt").write_text("\n".join(p.stem for p in gb_files))
    print(f"\nManifest written → {OUT_DIR}/manifest.txt")


if __name__ == "__main__":
    main()

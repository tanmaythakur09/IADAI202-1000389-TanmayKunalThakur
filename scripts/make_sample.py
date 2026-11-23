"""
make_sample.py - shrink the 794 MB SPARCS discharge file into the 5 % sample used by the app.

Why: the full file (2,101,588 rows) is far too big for GitHub (100 MB limit) and for
Streamlit Cloud's free memory, so the deployed app reads a random 5 % sample instead.

Usage (run from the repository root):
    python scripts/make_sample.py path/to/full_file.csv

Output:
    data/hospital_sample.csv.gz     about 105,000 rows, same 33 columns as the original

The file is read in chunks so it does not need much RAM, and the random seed is fixed so
anyone running this script gets exactly the same sample.
"""
import sys
from pathlib import Path

import pandas as pd

SRC = sys.argv[1]                                    # the big CSV
OUT = Path(__file__).resolve().parent.parent / "data" / "hospital_sample.csv.gz"
FRACTION = 0.05                                      # keep 5 % of the rows
CHUNK_ROWS = 200_000                                 # rows read at a time
SEED = 42                                            # makes the sample repeatable

parts, total = [], 0
for chunk in pd.read_csv(SRC, dtype=str, chunksize=CHUNK_ROWS, encoding_errors="replace"):
    total += len(chunk)
    parts.append(chunk.sample(frac=FRACTION, random_state=SEED))
    print(f"read {total:,} rows...", end="\r")

sample = pd.concat(parts, ignore_index=True)
OUT.parent.mkdir(exist_ok=True)
sample.to_csv(OUT, index=False, compression="gzip")
print(f"\nFull file: {total:,} rows | sample: {len(sample):,} rows | saved to {OUT} "
      f"({OUT.stat().st_size / 1e6:.1f} MB)")

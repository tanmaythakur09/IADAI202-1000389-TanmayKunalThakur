# data/

Place `hospital_sample.csv.gz` in this folder (the 5 % random sample of the full SPARCS 2021 inpatient
discharge file). The dashboard reads it from here.

The full 794 MB file is not stored in the repository because GitHub rejects files over 100 MB.
Re-create the sample with `python scripts/make_sample.py path/to/full_file.csv`.

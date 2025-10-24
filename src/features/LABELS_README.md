# Label Engineering Pipeline (Contact + BIP)

## Contact labels
python -m src.features.make_labels_contact --input data_raw --output data_proc/contact_labels.parquet

## Batted-ball labels (balls in play only)
python -m src.features.make_labels_bip --input data_raw --output data_proc/labels.parquet --bins data_proc/SxR_bins.json --S 10 --R 5

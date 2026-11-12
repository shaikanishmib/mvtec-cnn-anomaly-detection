# Experiment registry

Notebooks are the canonical chronological narrative. Before fitting, copy the
template in `configs/experiment_template.json`, fill its fields and call
`anomaly_detection.tracking.record`. Example IDs: `01-cnn-20261112-seed2026-start`
and `01-cnn-20261112-seed2026-result`. Write a distinct result or failure event
referencing the start ID in `training`; existing records cannot be overwritten.
Null values mean not yet measured, never zero performance. Preserve tracebacks,
OOMs, interrupted runs, rejected settings and interpretations. Meaningful config
changes require new IDs. Never rewrite history to retain only successes.

The entry hash covers the complete submitted record. Also record a dedicated
configuration hash, Git SHA (and dirty status), dataset/partition manifest hashes,
UTC start/end timestamps, package lock, checkpoint SHA256 and random seeds in
`training`/`environment`. Large tensors and raw predictions belong in ignored
directories; retain compact aggregate metrics and provenance here/reports.

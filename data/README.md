# Private data workspace

This directory contains placeholders only. Do not commit MIMIC records,
patient-level processed data, augmented data, synthetic data, checkpoints, or
run logs.

After obtaining authorized MIMIC access, follow the repository README to create
the model-ready files directly from the original MIMIC CSV tables:

- `Mimic3/real/real.csv`
- `Mimic4/real/real.csv`

The pipeline writes private augmented cohorts under
`Mimic*/augmented/<generator>/` and private synthetic cohorts under
`Mimic*/synthetic/<generator>/`.

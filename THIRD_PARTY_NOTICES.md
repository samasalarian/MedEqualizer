# Third-Party Notices

The following software is used by MedEqualizer under its original license. No
third-party implementation is relicensed as MedEqualizer-owned code.

## CTGAN 0.11.0

- Use: external Python dependency; its source is not bundled here
- Developer and licensor: DataCebo, Inc.
- Source: https://github.com/sdv-dev/CTGAN/tree/v0.11.0
- License: [Business Source License 1.1](https://github.com/sdv-dev/CTGAN/blob/v0.11.0/LICENSE)
- Change license: MIT, four years after the v0.11.0 release date

## HealthGAN `synthetic_data`

- Bundled location: `synthetic_data_generation/healthgan/synthetic_data/`
- Source: https://github.com/TheRensselaerIDEA/synthetic_data
- License: [MIT](synthetic_data_generation/healthgan/synthetic_data/LICENSE)
- Copyright: Copyright (c) 2020 TheRensselaerIDEA

## medGAN

- Bundled implementation: `synthetic_data_generation/medgan/medgan.py`
- Original implementation: Edward Choi (`mp2893`) and contributors
- Source: https://github.com/mp2893/medgan
- License: [BSD 3-Clause](synthetic_data_generation/medgan/LICENSE)
- Copyright: Copyright (c) 2017, mp2893

The surrounding MedEqualizer integration and workflow code is distinct from
these bundled implementations. The top-level MedEqualizer MIT License applies
only to MedEqualizer-owned code.

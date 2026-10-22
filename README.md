# MedEqualizer

MedEqualizer investigates and improves intersectional demographic representation
in synthetic healthcare data. It identifies underrepresented intersections of
gender, ethnicity, and age, augments them with quality-filtered synthetic records,
and evaluates representation and downstream mortality prediction. The manuscript
evaluates CTGAN, HealthGAN, and MedGAN on MIMIC-III and MIMIC-IV.

## Installation

Run commands from the repository root in bash.

| Conda environment | Purpose |
| --- | --- |
| `env` | CTGAN generation and augmentation, preprocessing, mortality prediction, Python bar charts |
| `fenv` | HealthGAN and MedGAN generation and augmentation; Python 3.6 / TensorFlow 1.13.1 on Linux x86_64 |

```bash
conda env create -f reproducibility/environment-env.yml
conda env create -f reproducibility/environment-fenv.yml
conda activate fenv
python -m pip install --no-deps -e ./synthetic_data_generation/healthgan/synthetic_data
```

### Visualization (Optional)

For visualization, install [R](https://cran.r-project.org/) and the required packages:

```bash
Rscript -e 'install.packages(c("optparse", "plotly", "formattable", "dplyr", "tidyr", "ggplot2", "lubridate", "rlang", "htmlwidgets", "plyr"), repos="https://cloud.r-project.org")'
```

R is only needed for the sunburst diagrams. To run augmentation and mortality
prediction without it, pass `--skip_sunburst` to `run_pipeline.py`.

## Data Preparation

MIMIC-III and MIMIC-IV require authorized access through PhysioNet and are not
included in this repository. Use these input layouts, replacing the example directories below with your own:

| Dataset | Files relative to the input directory |
| --- | --- |
| MIMIC-III | `PATIENTS.csv`, `ADMISSIONS.csv`, `DIAGNOSES_ICD.csv`, `ICUSTAYS.csv` |
| MIMIC-IV | `patients.csv`, `admissions.csv`, `diagnoses_icd.csv`, `icu/icustays.csv` (also accepts `.csv.gz`) |

```bash
conda activate env
python preprocessing/process_real_data.py \
  --input-dir /path/to/mimiciii --output data/Mimic3/real/real.csv
python preprocessing/process_real_data.py \
  --input-dir /path/to/mimiciv --output data/Mimic4/real/real.csv
```

Preprocessing creates model-ready `real.csv` files with these 11 columns:

```text
GENDER, ETHNICITY, AGE_GROUP, MORTALITY,
ADMISSION_LOCATION, ADMISSION_TYPE, INSURANCE, DISEASE,
FIRST_CAREUNIT, LAST_CAREUNIT, LOS
```

## Reproducing the Experiments

Execution order: preprocessing → baseline synthetic generation → MedEqualizer
augmentation/filtering → final synthetic generation → mortality prediction.
The pipeline performs augmentation, subgroup-specific one-class SVM filtering,
discriminative AUROC filtering, and final generation in one invocation.

The six final configurations are recorded in
[`reproducibility/final-configurations.json`](reproducibility/final-configurations.json).
Select one row at a time; repeat the following workflow for all six rows.

| Dataset | Model | tau | min_subgroup_size | epochs | aug_epochs | workers | max_tries | nodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mimic3 | ctgan | 12 | 5 | 300 | 1000 | 8 | 100 | 64 |
| mimic3 | healthgan | 14 | 5 | 10000 | 3000 | 8 | 300 | 64 |
| mimic3 | medgan | 8 | 5 | 1000 | 300 | 1 | 100 | 256 |
| mimic4 | ctgan | 12 | 5 | 300 | 1000 | 8 | 100 | 64 |
| mimic4 | healthgan | 8 | 5 | 10000 | 3000 | 8 | 300 | 64 |
| mimic4 | medgan | 14 | 5 | 1000 | 300 | 8 | 100 | 64 |

All runs use `nu=0.2`, `gamma=auto`, `auroc=0.9`,
`classifier=random_forest`, augmentation `batch_size=500`, `max_attempts=10`,
`n_samples=100000`, `aug_seed=42`, `gen_seed=42`, and `critic=5`.
HealthGAN and MedGAN augmentation executes serially. CTGAN does not use
`aug_epochs`; `nodes` and `critic` apply only to HealthGAN. MedGAN baseline and
final synthesis use 100 pretraining epochs and batch size 1000.

Set the variables from your selected row. This example selects MIMIC-III CTGAN;
use `cohort=Mimic4` for `dataset=mimic4`.

```bash
dataset=mimic3; cohort=Mimic3; model=ctgan
tau=12; epochs=300; aug_epochs=1000; workers=8; max_tries=100; nodes=64
real="data/$cohort/real/real.csv"
synth="data/$cohort/synthetic/$model"
mkdir -p "$synth"
```

Generate the baseline with the matching branch below. Preserve `before.csv`
before running augmentation and final generation.

```bash
case "$model" in
  ctgan)
    conda activate env
    python synthetic_data_generation/ctgan/ctgan_gen.py \
      --data "$real" --version "$dataset" --epochs "$epochs" \
      --n_samples 100000 --seed 42 --output_path "$synth/before.csv" \
      --model_output_path "synthetic_data_generation/ctgan/gen_data/$dataset/ctgan_model.pkl"
    ;;
  healthgan)
    conda activate fenv
    python synthetic_data_generation/healthgan/healthgan_gen.py \
      --data "$real" --version "$dataset" --epochs "$epochs" \
      --nodes "$nodes" --critic 5 --n_samples 100000 --seed 42
    cp "synthetic_data_generation/healthgan/gen_data/$dataset/synthetic_decoded.csv" "$synth/before.csv"
    ;;
  medgan)
    conda activate fenv
    python synthetic_data_generation/medgan/medgan_gen.py \
      --data "$real" --version "$dataset" --epochs "$epochs" \
      --pretrain_epochs 100 --batch_size 1000 --n_samples 100000 --seed 42
    cp "synthetic_data_generation/medgan/gen_data/$dataset/synthetic_decoded.csv" "$synth/before.csv"
    ;;
esac
```

In the same shell and generator environment, run augmentation, filtering, final
synthesis, and R sunbursts. `--force_augment` ensures a fresh augmentation run if its output already exists.
If no candidates pass the quality gates, the pipeline stops before final synthesis.

```bash
python run_pipeline.py --dataset "$dataset" --model "$model" --tau "$tau" \
  --nu 0.2 --gamma auto --auroc 0.9 --classifier random_forest \
  --workers "$workers" --batch_size 500 --max_tries "$max_tries" --max_attempts 10 \
  --aug_epochs "$aug_epochs" --min_subgroup_size 5 --epochs "$epochs" \
  --n_samples 100000 --nodes "$nodes" --critic 5 --aug_seed 42 --gen_seed 42 \
  --force_augment --skip_barchart

For execution without visualization, add `--skip_sunburst` to the pipeline
command and omit the subsequent bar-chart command.

After generating all six baseline/final dataset pairs, run mortality prediction
in `env`:

```bash
conda activate env
python pred/mortality_classification.py
```

The driver uses stratified 80/20 splits with seeds `42 52 62 72 82` and reports
AUROC for Decision Tree, KNN, Gradient Boosting, Histogram Gradient Boosting,
and MLP.

## Outputs

| Output | Location |
| --- | --- |
| Model-ready input | `data/<Mimic3-or-Mimic4>/real/real.csv` |
| Augmented cohort | `data/<cohort>/augmented/<model>/<tau>_0p2_auto_0p9_random_forest.csv` |
| Baseline / final synthetic data | `data/<cohort>/synthetic/<model>/before.csv` / `after_tau=<tau>_nu0p2_auto_0p9_random_forest.csv` |
| Mortality evaluation | `pred/mortality_auroc_table.csv` |
| Parameters, equity CSVs, sunburst HTML, bar-chart PNG | `visualization/results/<model>/<tag>/` |

Generator intermediates and checkpoints are saved under
`synthetic_data_generation/<model>/`; augmentation logs are saved as
`data/augmentation_counts_<model>.xlsx`. Sunburst HTML requires its sibling
`*_files/` assets.

Keep MIMIC data, patient-level processed/augmented/synthetic records, encoded
intermediates, trained models, checkpoints, and run logs private; do not upload
them publicly. Review aggregate results and figures for disclosure before release.

## Licensing

MedEqualizer-owned code is licensed under the [MIT License](LICENSE).

CTGAN 0.11.0 is an external dependency licensed by DataCebo under the
[Business Source License 1.1](https://github.com/sdv-dev/CTGAN/blob/v0.11.0/LICENSE).
The bundled HealthGAN package retains its original
[MIT license](synthetic_data_generation/healthgan/synthetic_data/LICENSE), and
the bundled medGAN implementation retains its original
[BSD 3-Clause license](synthetic_data_generation/medgan/LICENSE). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and scope.

## Citation

If you use MedEqualizer in your research, please cite:

Salarian, S., Zhang, Y., Padhee, S., & Parthasarathy, S. (2025).
*MedEqualizer: A Framework Investigating Bias in Synthetic Medical Data and Mitigation via Augmentation.*
arXiv:2511.01054. https://doi.org/10.48550/arXiv.2511.01054

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

R is only needed for the sunburst diagrams. 

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
FIRST_CAREUNIT, LAST_CAREUNIT, LOS.
```

## Reproducing the Experiments

Execution order: preprocessing → baseline synthetic generation → MedEqualizer
augmentation/filtering → final synthetic generation. Downstream mortality
prediction (TRTR/TSTR) is a separate, leakage-free pipeline described below.
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
  --force_augment 
```

For execution without visualization, add `--skip_sunburst` and '--skip_barchart' to the pipeline command.

### Random oversampling (ROS) baseline

`ros_baseline.py` compares MedEqualizer with random oversampling (ROS) on
representation fairness, for all six final configurations. It requires the
augmented cohorts and the `before.csv` / `after_tau=...` synthetic data
produced above.

For each configuration, ROS keeps the real rows of the MedEqualizer augmented
cohort and, in every gender × ethnicity × age-group subgroup, duplicates
exactly as many real records (sampled with replacement) as MedEqualizer added
there. The same generator is then retrained on the ROS cohort with the
finalized hyperparameters and 100,000 synthetic records. 

```bash
conda activate env
python ros_baseline.py augment --data_dir data            # ROS cohorts
for i in 0 1 2 3 4 5; do python ros_baseline.py generate --data_dir data --config $i; done
python ros_baseline.py evaluate --data_dir data           # results/ros/ros_comparison.csv
python ros_baseline.py plot                               # results/ros/ros_vs_medequalizer.pdf
```

`generate` calls the `fenv` interpreter for HealthGAN and MedGAN, and
`evaluate` calls `Rscript` from `fenv`. `evaluate` labels the 80
mortality × race × age × gender subgroups of the Original, ROS, and
MedEqualizer synthetic data with the same logarithmic disparity metric as the
sunburst diagrams, and reports the count of subgroups in each representation
category. Run `python ros_baseline.py selftest` to validate the ROS and
evaluation steps on artificial data.

### Mortality prediction (TRTR / TSTR)

Mortality prediction uses a separate pipeline so that held-out real test
patients never reach augmentation, filtering, or GAN training. For each
dataset, one patient-level, outcome-stratified 80/20 split is made (seed 42);
MedEqualizer augmentation and GAN training use only the training split, with
the finalized configurations and GAN seeds `42 52 62 72 82`.

```bash
conda activate env
export CUDA_VISIBLE_DEVICES=""
python pred/mortality_data.py split --data_dir data
for i in 0 1 2 3 4 5; do python pred/mortality_data.py augment --config $i; done
for k in $(seq 0 59); do python pred/mortality_data.py generate --task $k; done
python pred/mortality_data.py verify
python pred/mortality_cls.py
```

`generate` runs one of 60 trainings (6 configurations × original/MedEqualizer ×
5 seeds) and can be parallelized, e.g. as a job array over `--task`.
HealthGAN and MedGAN steps call the `fenv` interpreter automatically.

`mortality_cls.py` trains KNN, random forest, and extra trees on the real
training split (TRTR) and on each synthetic dataset (TSTR), and reports AUROC
on the held-out real test patients: TRTR with a 95% bootstrap CI over test
patients, TSTR as the mean over five GAN runs with a t-based 95% CI (df = 4).
The outcome is any recorded death (`MORTALITY = 1`), not in-hospital or
30-day mortality.

## Outputs

| Output | Location |
| --- | --- |
| Model-ready input | `data/<Mimic3-or-Mimic4>/real/real.csv` |
| Augmented cohort | `data/<cohort>/augmented/<model>/<tau>_0p2_auto_0p9_random_forest.csv` |
| Baseline / final synthetic data | `data/<cohort>/synthetic/<model>/before.csv` / `after_tau=<tau>_nu0p2_auto_0p9_random_forest.csv` |
| ROS baseline (representation fairness) | `results/ros/` |
| Mortality data (splits, augmented cohorts, synthetic data) | `results/tstr_trtr/` |
| Mortality evaluation (AUROC, 95% CI) | `results/mortality_cls_auroc.csv` |
| Parameters, equity CSVs, sunburst HTML, bar-chart PNG | `visualization/results/<model>/<tag>/` |

Generator intermediates and checkpoints are saved under
`synthetic_data_generation/<model>/`; augmentation logs are saved as
`data/augmentation_counts_<model>.xlsx`. Sunburst HTML requires its sibling
`*_files/` assets.

Keep MIMIC data, patient-level processed/augmented/synthetic records
(including the real train/test splits and synthetic data under `results/`),
encoded intermediates, trained models, checkpoints, and run logs private; do
not upload them publicly. Review aggregate results and figures for disclosure before release.

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

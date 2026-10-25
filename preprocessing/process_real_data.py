#!/usr/bin/env python3
"""
MIMIC Data Processing Script (MIMIC-III and MIMIC-IV compatible)

This script processes MIMIC medical data by merging patient information,
admission details, diagnoses, and ICU stay data to create a comprehensive
dataset for analysis.

Required input files:
    - PATIENTS.csv
    - ADMISSIONS.csv
    - DIAGNOSES_ICD.csv
    - ICUSTAYS.csv

Output:
    - A model-ready real.csv containing the shared MIMIC-III/MIMIC-IV feature
      contract used by the release pipeline.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path


AGE_GROUPS = {"<= 45", "46-65", "66-80", "+81"}

# Shared release input contract. Each retained field is either an equity
# attribute, the mortality target, or a feature explicitly constructed by this
# script and consumed by the existing generator/evaluation code. Raw MIMIC
# identifiers, dates, and age fields are preprocessing intermediates only.
FINAL_MODEL_COLUMNS = [
    'GENDER',
    'ETHNICITY',
    'AGE_GROUP',
    'MORTALITY',
    'ADMISSION_LOCATION',
    'ADMISSION_TYPE',
    'INSURANCE',
    'DISEASE',
    'FIRST_CAREUNIT',
    'LAST_CAREUNIT',
    'LOS',
]

INPUT_FILE_CANDIDATES = {
    'patients': [
        'PATIENTS.csv', 'PATIENTS.csv.gz',
        'patients.csv', 'patients.csv.gz',
        'hosp/patients.csv', 'hosp/patients.csv.gz',
    ],
    'admissions': [
        'ADMISSIONS.csv', 'ADMISSIONS.csv.gz',
        'admissions.csv', 'admissions.csv.gz',
        'hosp/admissions.csv', 'hosp/admissions.csv.gz',
    ],
    'diagnoses': [
        'DIAGNOSES_ICD.csv', 'DIAGNOSES_ICD.csv.gz',
        'diagnoses_icd.csv', 'diagnoses_icd.csv.gz',
        'hosp/diagnoses_icd.csv', 'hosp/diagnoses_icd.csv.gz',
    ],
    'icustays': [
        'ICUSTAYS.csv', 'ICUSTAYS.csv.gz',
        'icustays.csv', 'icustays.csv.gz',
        'icu/icustays.csv', 'icu/icustays.csv.gz',
    ],
}


def resolve_input_file(input_dir, table):
    """Find a required table in flat MIMIC-III or standard MIMIC-IV layout."""
    for relative_path in INPUT_FILE_CANDIDATES[table]:
        candidate = input_dir / relative_path
        if candidate.is_file():
            return candidate
    searched = ', '.join(INPUT_FILE_CANDIDATES[table])
    raise FileNotFoundError(
        f"Could not find the {table} table under {input_dir}. Tried: {searched}"
    )


def create_age_group(age_values):
    """Categorize AGE using the manuscript boundaries, preserving missingness."""
    age = pd.to_numeric(age_values, errors='coerce')
    invalid = age_values.notna() & (age.isna() | ~np.isfinite(age))
    if invalid.any():
        raise ValueError("AGE contains nonnumeric or non-finite non-missing values")

    age_group = pd.Series(pd.NA, index=age_values.index, dtype='string')
    age_group.loc[age <= 45] = '<= 45'
    age_group.loc[(age > 45) & (age <= 65)] = '46-65'
    age_group.loc[(age > 65) & (age <= 80)] = '66-80'
    age_group.loc[age > 80] = '+81'

    if not age_group[age.notna()].notna().all():
        raise ValueError("At least one non-missing AGE was not categorized")
    if not age_group[age.isna()].isna().all():
        raise ValueError("Missing AGE values were not preserved as missing")
    if not set(age_group.dropna().unique()).issubset(AGE_GROUPS):
        raise ValueError("Unexpected AGE_GROUP label produced")
    return age_group


def select_model_features(df):
    """Return the common model-ready feature contract for both MIMIC versions."""
    result = df.copy()
    result['AGE_GROUP'] = create_age_group(result['AGE'])

    missing = [column for column in FINAL_MODEL_COLUMNS if column not in result]
    if missing:
        raise ValueError(f"Required model columns are missing: {missing}")

    return result[FINAL_MODEL_COLUMNS].copy()


def detect_mimic_version(patients_df):
    """
    Detect MIMIC version based on column names.
    
    Args:
        patients_df: Patients dataframe
        
    Returns:
        str: 'mimic3' or 'mimic4'
    """
    if 'SUBJECT_ID' in patients_df.columns:
        return 'mimic3'
    elif 'subject_id' in patients_df.columns:
        return 'mimic4'
    else:
        raise ValueError("Cannot determine MIMIC version from column names")


def categorize_patient_disease(icd_codes, icd_versions=None):
    """
    Categorize patients based on their ICD diagnosis codes.
    Supports both ICD-9 and ICD-10.
    
    Args:
        icd_codes: List of ICD codes for a patient
        icd_versions: List of ICD versions (9 or 10) - only for MIMIC-IV
        
    Returns:
        str: Disease category - "Any malignancy", "Congestive heart failure", 
             "BOTH", or "OTHER"
    """
    # ICD-9 codes
    malignancy_codes_icd9 = range(140, 240)
    heart_failure_codes_icd9 = {428}
    
    # ICD-10 codes (simplified mapping)
    malignancy_prefixes_icd10 = ['C', 'D0', 'D1', 'D2', 'D3', 'D4']  # C00-D49
    heart_failure_codes_icd10 = ['I50']  # I50.x
    
    has_malignancy = False
    has_heart_failure = False
    
    for i, code in enumerate(icd_codes):
        if pd.isna(code):
            continue
            
        code_str = str(code).strip()
        
        # Determine if this is ICD-9 or ICD-10
        if icd_versions is not None:
            version = icd_versions[i]
        else:
            # MIMIC-III - assume ICD-9
            version = 9
        
        if version == 9:
            # ICD-9 processing
            if code_str and code_str[0].isdigit() and len(code_str) >= 3:
                try:
                    code_prefix = int(code_str[:3])
                    if code_prefix in malignancy_codes_icd9:
                        has_malignancy = True
                    if code_prefix in heart_failure_codes_icd9:
                        has_heart_failure = True
                except ValueError:
                    continue
        else:
            # ICD-10 processing
            if any(code_str.startswith(prefix) for prefix in malignancy_prefixes_icd10):
                has_malignancy = True
            if any(code_str.startswith(prefix) for prefix in heart_failure_codes_icd10):
                has_heart_failure = True
    
    if has_malignancy and has_heart_failure:
        return "BOTH"
    elif has_malignancy:
        return "Any malignancy"
    elif has_heart_failure:
        return "Congestive heart failure"
    else:
        return "OTHER"


def process_mimic_data(input_dir='.', output_file='processed_real_data.csv'):
    """
    Process MIMIC data files (III or IV) and create a comprehensive dataset.
    
    Args:
        input_dir: Directory containing input CSV files
        output_file: Output CSV filename
    """
    input_path = Path(input_dir)
    
    print("Loading data files...")
    # Load datasets
    patients = pd.read_csv(resolve_input_file(input_path, 'patients'))
    admissions = pd.read_csv(resolve_input_file(input_path, 'admissions'))
    diagnoses = pd.read_csv(resolve_input_file(input_path, 'diagnoses'))
    icustays = pd.read_csv(resolve_input_file(input_path, 'icustays'))
    
    # Detect MIMIC version
    version = detect_mimic_version(patients)
    print(f"Detected {version.upper()}")
    
    # Normalize column names based on version
    if version == 'mimic4':
        # Standardize to uppercase for consistency
        patients.columns = patients.columns.str.upper()
        admissions.columns = admissions.columns.str.upper()
        diagnoses.columns = diagnoses.columns.str.upper()
        icustays.columns = icustays.columns.str.upper()
        
        # Map MIMIC-IV specific column names to MIMIC-III equivalents
        admissions.rename(columns={'RACE': 'ETHNICITY'}, inplace=True)
        diagnoses.rename(columns={'ICD_CODE': 'ICD9_CODE'}, inplace=True)
    
    print("Processing patient data...")
    # Create mortality indicator
    patients['MORTALITY'] = patients['DOD'].apply(
        lambda x: 0 if pd.isna(x) else 1
    )
    
    # Handle age based on version
    if version == 'mimic3':
        # Convert date columns to datetime
        patients['DOB'] = pd.to_datetime(patients['DOB'])
        admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
        
        print("Merging patients and admissions...")
        # Merge admissions with patients
        merged = pd.merge(admissions, patients, on='SUBJECT_ID', how='inner')
        
        # Calculate age
        merged['AGE'] = (merged['ADMITTIME'].dt.year - merged['DOB'].dt.year)
        # Cap age at 90 (MIMIC privacy requirement)
        merged.loc[merged['AGE'] > 89, 'AGE'] = 90
        
        print("Computing maximum age per patient...")
        # Get maximum age per patient
        max_age = merged.groupby('SUBJECT_ID')['AGE'].max().reset_index()
        
    else:  # mimic4
        # MIMIC-IV uses anchor_age - use this directly
        print("Using anchor_age from MIMIC-IV...")
        max_age = patients[['SUBJECT_ID', 'ANCHOR_AGE']].copy()
        max_age.rename(columns={'ANCHOR_AGE': 'AGE'}, inplace=True)
        
        # Convert admittime for merging
        admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
        
        # Merge for getting admission details
        merged = pd.merge(admissions, patients, on='SUBJECT_ID', how='inner')
    
    # Merge patients with age
    patients_with_age = pd.merge(patients, max_age, on='SUBJECT_ID', how='inner')
    
    print("Extracting admission details...")
    # Get first admission details per patient
    ethnicity_and_related_values = merged.groupby('SUBJECT_ID')[[
        'HADM_ID', 'ETHNICITY', 'ADMISSION_LOCATION', 
        'ADMISSION_TYPE', 'INSURANCE'
    ]].first().reset_index()
    
    # Merge to create patients_admission dataset
    patients_admission = pd.merge(
        patients_with_age, 
        ethnicity_and_related_values, 
        on='SUBJECT_ID', 
        how='inner'
    )
    
    print("Processing diagnoses...")
    # Merge with diagnoses
    patient_diagnoses = pd.merge(
        patients_admission, 
        diagnoses, 
        on='SUBJECT_ID', 
        how='inner'
    )
    
    # Categorize diseases by patient
    if version == 'mimic4' and 'ICD_VERSION' in diagnoses.columns:
        # MIMIC-IV has ICD versions
        print("Processing ICD-9 and ICD-10 codes...")
        disease_categories = (
            patient_diagnoses.groupby('SUBJECT_ID')
            .apply(lambda x: categorize_patient_disease(
                x['ICD9_CODE'].tolist(), 
                x['ICD_VERSION'].tolist()
            ))
            .reset_index(name='DISEASE')
        )
    else:
        # MIMIC-III has only ICD-9
        print("Processing ICD-9 codes...")
        disease_categories = (
            patient_diagnoses.groupby('SUBJECT_ID')['ICD9_CODE']
            .apply(list)
            .apply(lambda x: categorize_patient_disease(x, None))
            .reset_index(name='DISEASE')
        )
    
    # Merge disease categories back
    patient_admission_diagnosis = patients_admission.merge(
        disease_categories, 
        on='SUBJECT_ID', 
        how='left'
    )
    
    print("Processing ICU stay data...")
    # Get first ICU stay details per patient
    icustays_filt = icustays.groupby('SUBJECT_ID')[[
        'FIRST_CAREUNIT', 'LAST_CAREUNIT', 'LOS'
    ]].first().reset_index()
    
    # Merge with ICU stays
    patient_admission_diagnosis_icustays = patient_admission_diagnosis.merge(
        icustays_filt, 
        on='SUBJECT_ID', 
        how='inner'
    )
    
    print("Filtering and selecting model features...")
    # Filter out newborns (age == 0) and missing LOS
    final_df = patient_admission_diagnosis_icustays[
        patient_admission_diagnosis_icustays['AGE'] != 0
    ]
    final_df = final_df.dropna(subset=['LOS'])

    # AGE is retained only long enough to derive AGE_GROUP. The resulting
    # dataframe intentionally excludes raw identifiers, dates, and other
    # preprocessing-only patient-table fields.
    final_df = select_model_features(final_df)
    
    print(f"\n{'='*60}")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"{'='*60}")
    print(f"\nColumn names:\n{list(final_df.columns)}")
    print(f"\nMissing values summary:")
    missing = final_df.isna().sum()
    print(missing[missing > 0] if (missing > 0).any() else "No missing values!")
    print(f"\nDisease distribution:")
    print(final_df['DISEASE'].value_counts())
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print("✓ Processing complete!")
    
    return final_df


def main():
    parser = argparse.ArgumentParser(
        description='Process MIMIC-III or MIMIC-IV data files into a comprehensive dataset'
    )
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default='.', 
        help='Directory containing input CSV files (default: current directory)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='processed_real_data.csv',
        help='Output CSV filename (default: processed_real_data.csv)'
    )
    
    args = parser.parse_args()
    
    try:
        process_mimic_data(args.input_dir, args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUse either the flat MIMIC-III CSV layout or the standard "
              "MIMIC-IV hosp/ and icu/ layout. Uncompressed .csv and "
              "compressed .csv.gz files are supported.")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

# python preprocessing/process_real_data.py --input-dir ./data/Mimic4/raw --output ./data/processed_mimic4.csv

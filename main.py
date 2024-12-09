import ipywidgets as widgets
import sys
from pathlib import Path
import os
import importlib


module_path='preprocessing/day_intervals_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)

module_path='utils'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='preprocessing/hosp_module_preproc'
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path='model'
if module_path not in sys.path:
    sys.path.append(module_path)
#print(sys.path)
root_dir = os.path.dirname(os.path.abspath('UserInterface.ipynb'))
import day_intervals_cohort
from day_intervals_cohort import *

import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

import data_generation_icu

import data_generation
import evaluation

import feature_selection_hosp
from feature_selection_hosp import *

# import train
# from train import *


import ml_models
from ml_models import *

import dl_train
from dl_train import *

import tokenization
from tokenization import *


import behrt_train
from behrt_train import *

import feature_selection_icu
from feature_selection_icu import *
import fairness
import callibrate_output

importlib.reload(day_intervals_cohort)
import day_intervals_cohort
from day_intervals_cohort import *

importlib.reload(day_intervals_cohort_v2)
import day_intervals_cohort_v2
from day_intervals_cohort_v2 import *

importlib.reload(data_generation_icu)
import data_generation_icu
importlib.reload(data_generation)
import data_generation

importlib.reload(feature_selection_hosp)
import feature_selection_hosp
from feature_selection_hosp import *

importlib.reload(feature_selection_icu)
import feature_selection_icu
from feature_selection_icu import *

importlib.reload(tokenization)
import tokenization
from tokenization import *

importlib.reload(ml_models)
import ml_models
from ml_models import *

importlib.reload(dl_train)
import dl_train
from dl_train import *

importlib.reload(behrt_train)
import behrt_train
from behrt_train import *

importlib.reload(fairness)
import fairness

importlib.reload(callibrate_output)
import callibrate_output

importlib.reload(evaluation)
import evaluation


#version = 'Version 1'
#input4 = 'Phenotype'
#input2 = 'Heart Failure in 30 days'
data_icu='ICU'
icd_code='I50'
disease_label='I50'
input1 = 'ICU'
#input3 = 'No Disease Filter'



label='Readmission'
time=30

data_mort=label=="Mortality"
data_admn=label=='Readmission'
data_los=label=='Length of Stay'

version_path="mimiciv/1.0"
cohort_output = day_intervals_cohort.extract_data(input1.value,label,time,icd_code, root_dir,disease_label)

print("Feature Selection")


if data_icu:
    
    diag_flag = True  # Diagnosis
    out_flag = True   # Output Events
    chart_flag = True # Chart Events (Labs and Vitals)
    proc_flag = True  # Procedures
    med_flag = True   # Medications

    feature_icu(cohort_output, version_path, diag_flag, out_flag, chart_flag, proc_flag, med_flag)


else:
    
    diag_flag = True  # Diagnosis
    lab_flag = True   # Labs
    proc_flag = True  # Procedures
    med_flag = True   # Medications

    feature_nonicu(cohort_output, version_path, diag_flag, lab_flag, proc_flag, med_flag)

print("**Feature selection process completed.**")

print("Feature Preprocessing")


group_diag = 'Convert ICD-9 to ICD-10 and group ICD-10 codes'
group_med = 'Yes'  # Group Medication codes
group_proc = 'ICD-10'  # Keep only ICD-10 procedure codes

if data_icu:
    if diag_flag:
        print(f"Grouping ICD-10 diagnosis codes: {group_diag}")
    preprocess_features_icu(
        cohort_output,
        diag_flag=diag_flag,
        group_diag=group_diag,
        group_med=False,
        group_proc=False,
        chart_flag=False,
        time_series_length=0,
        time_bucket_size=0
    )
else:
    if diag_flag:
        print(f"Grouping ICD-10 diagnosis codes: {group_diag}")
    if med_flag:
        print(f"Grouping medication codes to non-proprietary names: {group_med}")
    if proc_flag:
        print(f"Keeping only ICD-10 procedure codes: {group_proc}")
    preprocess_features_hosp(
        cohort_output,
        diag_flag=diag_flag,
        proc_flag=proc_flag,
        med_flag=med_flag,
        chart_flag=False,
        group_diag=group_diag,
        group_med=group_med,
        group_proc=group_proc,
        lab_flag=False,
        fairness_flag=False,
        time_series_length=0,
        time_bucket_size=0
    )

print("**Feature preprocessing completed.**")

if data_icu:
    generate_summary_icu(diag_flag,proc_flag,med_flag,out_flag,chart_flag)
else:
    generate_summary_hosp(diag_flag,proc_flag,med_flag,lab_flag)

print("Feature Selection")


select_diag = True  # Select Diagnosis features
select_med = True   # Select Medication features
select_proc = True  # Select Procedures features
select_lab = True   # Select Labs features (for non-ICU)
select_out = True   # Select Output Event features (for ICU)
select_chart = True # Select Chart Event features (for ICU)

if data_icu:
    print("Performing feature selection for ICU data...")
    print(f"Diagnosis: {select_diag}, Medications: {select_med}, Procedures: {select_proc}, Output Events: {select_out}, Chart Events: {select_chart}")
    features_selection_icu(
        cohort_output, 
        diag_flag=True,
        proc_flag=True,
        med_flag=True,
        out_flag=True,
        chart_flag=True,
        select_diag=select_diag,
        select_med=select_med,
        select_proc=select_proc,
        select_out=select_out,
        select_chart=select_chart
    )
else:
    print("Performing feature selection for non-ICU data...")
    print(f"Diagnosis: {select_diag}, Medications: {select_med}, Procedures: {select_proc}, Labs: {select_lab}")
    features_selection_hosp(
        cohort_output,
        diag_flag=True,
        proc_flag=True,
        med_flag=True,
        lab_flag=True,
        select_diag=select_diag,
        select_med=select_med,
        select_proc=select_proc,
        select_lab=select_lab
    )

print("**Feature selection process completed.**")

print("Feature Preprocessing: Outlier Removal")


if data_icu:
    if chart_flag:
        print("Outlier removal for chart events:")
        clean_chart = True  # Enable outlier cleaning
        impute_outlier_chart = True  # Impute outliers instead of removing
        thresh = 98  # Right outlier threshold
        left_thresh = 0  # Left outlier threshold
        
        print(f"Cleaning: {clean_chart}, Imputation: {impute_outlier_chart}, Threshold: {thresh}, Left Threshold: {left_thresh}")
        preprocess_features_icu(
            cohort_output, 
            False,  # diag_flag
            False,  # med_flag
            chart_flag,
            clean_chart,
            impute_outlier_chart,
            thresh,
            left_thresh
        )
else:
    if lab_flag:
        print("Outlier removal for lab events:")
        clean_lab = True  # Enable outlier cleaning
        impute_outlier = True  # Impute outliers instead of removing
        thresh = 98  # Right outlier threshold
        left_thresh = 0  # Left outlier threshold
        
        print(f"Cleaning: {clean_lab}, Imputation: {impute_outlier}, Threshold: {thresh}, Left Threshold: {left_thresh}")
        preprocess_features_hosp(
            cohort_output, 
            False,  # diag_flag
            False,  # proc_flag
            False,  # med_flag
            lab_flag,
            False,  # group_diag
            False,  # group_med
            False,  # group_proc
            clean_lab,
            impute_outlier,
            thresh,
            left_thresh
        )

print("**Outlier removal and preprocessing completed.**")

print("=======Time-series Data Representation=======")


include = 72  # Include last 72 hours
bucket = 2    # Use 2-hour buckets
impute = 'Mean'  # Forward fill with mean imputation
predW = 2     # Prediction window length (set to 2 hours for mortality task if applicable)

print(f"Time-series Parameters: Include: {include} hours, Bucket Size: {bucket} hours, Imputation: {impute}, Prediction Window: {predW} hours")


if data_icu:
    print("Generating time-series data for ICU cohort...")
    gen = data_generation_icu.Generator(
        cohort_output, 
        data_mort, 
        data_admn, 
        data_los, 
        diag_flag, 
        proc_flag, 
        out_flag, 
        chart_flag, 
        med_flag, 
        impute, 
        include, 
        bucket, 
        predW
    )
else:
    print("Generating time-series data for non-ICU cohort...")
    gen = data_generation.Generator(
        cohort_output, 
        data_mort, 
        data_admn, 
        data_los, 
        diag_flag, 
        lab_flag, 
        proc_flag, 
        med_flag, 
        impute, 
        include, 
        bucket, 
        predW
    )

print("**Time-series representation and data generation completed.**")

print("Model Training Configuration")


selected_model = 'LSTM Attention GNN'  # Model choice
cv = 0  # No cross-validation
oversampling = True  # Enable oversampling for the minority class

print(f"Selected Model: {selected_model}")
print(f"Cross-Validation: {'No CV' if cv == 0 else f'{cv}-fold CV'}")
print(f"Oversampling for minority class: {oversampling}")

# Model training
if data_icu:
    print("Training model for ICU data...")
    model = dl_train.DL_models(
        data_icu=data_icu,
        diag_flag=diag_flag,
        proc_flag=proc_flag,
        out_flag=out_flag,
        chart_flag=chart_flag,
        med_flag=med_flag,
        lab_flag=False,
        model_type=selected_model,
        cv=cv,
        oversampling=oversampling,
        model_name='attn_icu_read',
        train=True
    )
else:
    print("Training model for non-ICU data...")
    model = dl_train.DL_models(
        data_icu=data_icu,
        diag_flag=diag_flag,
        proc_flag=proc_flag,
        out_flag=False,
        chart_flag=False,
        med_flag=med_flag,
        lab_flag=lab_flag,
        model_type=selected_model,
        cv=cv,
        oversampling=oversampling,
        model_name='attn_icu_read',
        train=True
    )

print("**Model training process completed.**")

fairness.fairness_evaluation(inputFile='outputDict',outputFile='fairnessReport')
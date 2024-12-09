
![dict](https://github.com/user-attachments/assets/4e209ecf-782b-4421-adb5-1514ab725c1d)

## Files

- **data_generation.py** and **data_generation_icu.py**
	are the files that create smooth time-series representation from the options 
  The output is saved in csv and dictionary format.
  
- **evaluation.py**
  contains code to perform evaluations on predictions made by model.
 
  
- **fairness.py**
  contains code to perform fairness evaluations on predictions made by model.
  
  
- **parameters.py**
  contains the list of hyperparameters for deep learning models.
  
 
  
- **mimic_model.py**
  contains definition of deep learning models included in the pipeline.
  
- **dl_train.py**
  contains the code to train and test deep learning models included in the pipeline.
 
### Preprocessing
- **./day_intervals_preproc**
  - **day_intervals_cohort.py** file is used to extract samples, labels and demographic data for cohorts.
  - **disease_cohort.py** is used to filter samples based on diagnoses codes at time of admission
  
- **./hosp_module_preproc**
  - **feature_selection_hosp.py** is used to extract, clean and summarize selected features for non-ICU data.
  - **feature_selection_icu.py** is used to extract, clean and summarize selected features for ICU data.
  Both above files internally use files in /./utils folder for feature extraction, cleaning, outlier removal, unit conversion.
  
  ### Utils

- **hosp_preprocess_util.py** and **icu_preprocess_util.py**
  These files are used to read original feature csv files downloaded from MIMIC-IV and clean (removing NAs, removing duplicates, etc) and
  save feature files for the selected cohort in ./data/features folder.
  
- **outlier_removal.py**
  removes outlier or imputes outlier with outlier threshold values.
  
- **uom_conversion.py**
  unit conversion to highest frequency unit for each itemid in labevents and chartevents data
  
- **labs_preprocess_util.py**
  finds the missing admission ids in labevents data by placinf timestamp of labevent between the admission and discharge time of the admission for the patient.

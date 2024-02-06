# TruEra Diagnostics
## Project Creation & Ingestion Demonstration: scripting with the TruEra Python SDK

## Four TruEra Python SDK demo scripts - TL;DR
1. Script 1 demonstrates TruEra project creation _leveraging a trained model object and input data_
2. Script 2 persist previously resources in a TruEra project for external use
3. Script 3 merges data from (1) and (2) in preparation to demonstrate data only / "virtual" model ingestion
4. Script 4 uses outputs of script (3) to create and a new TruEra project and populate the ML observability metrics within it, _without using the underlying model object_

### 1. truera_ingestion_1_create_project_dev_preds_FI.py
This script creates a new project and uses a pre-trained model object to generate predictions & feature influences, which are then added to an existing data split

- Inputs: 
    - TruEra credential (token)
    - TruEra deployment URL
    - pre_train.csv
    - post_train.csv
    - labels_train.csv
    - rf_v1.pkl
- Outputs:
    - feature_map.pkl (This is optional. It is used for mapping post transform model features to pre transform model features to simplify analysis in TruEra project)
        
        
### 2. truera_ingestion_1_create_project_dev_preds_FI.py
This script retrieves predictions & feature influences from the project that was created with script 1. It is intended to demonstrate how to retrieve pre-computed data from a TruEra project, and to persist it for external use / other purposes. 

- Inputs: 
    - TruEra credential (token)
    - TruEra deployment URL
- Outputs: CSVs containing predictions & feature influences for existing TruEra project's model and one data split (training data)
    - rf_train_preds.csv
    - rf_train_FIs.csv
        

### 3. truera_ingestion_3_data_merge.py
This script merges the training data (inputs) that were also used in script 1, with the retrieved predictions and feature influences persisted to CSVs in script 2

This is less a demonstration of TruEra, than of an example of how one might use the TruEra utility function merge_dataframes_and_create_column_spec to merge data in various files/tables, to prepare for data only "virtual model" ingestion.

This script also demonstrates the ability to save and persist more than one column spec -- consider a scenario where ones data may share a model input (feature) schema, but some data (e.g., production data) may not have labels available, and/or predictions, feature influences, or various other types of extra data, at the time that it is ingested into a project. 

- Inputs: 
    - pre_train.csv
    - post_train.csv
    - labels_train.csv
    - rf_train_preds.csv
    - rf_train_FIs.csv
    
- Outputs:
    - rf_train_data_df.csv
    - column_spec.pkl: for training data -- column spec includes feature influences
    - background_data_df.csv: background data; for virtual model ingestion requirement
    - background_column_spec.pkl: for background split -- column spec does not include feature influences. Optionally, we include predictions and labels. These can be omitted from a background split if not available (e.g., if using a pre-defined split as a background split whose labels and predictions are not readily available). 
    
 
### 4. truera_ingestion_4_data_only_virtual_model_ingestion.py
This script leverages artifacts produced in scripts 2 & 3 to create a new TruEra project using the data only "virtual model" approach. 

Note that, here, we used script 1 and 2 to generate the predictions and feature influences that were later used, in conjunction with the same data from script 1, to create a _new project_ that contains the same results as the projected created in step 1, _without utilizing the model object_. 

In some real world scenario, one may retrieve predictions and/or feature influences from an arbitrary external location, to achieve this same purpose. We just happen to have access to the model object to create these demonstration scripts, and thus used it to generate the model outputs that were used later on. 

- Inputs:
    - background_data_df.csv
    - rf_train_data_df.csv
    - column_spec.pkl
    - background_column_spec.pkl
    - feature_map.pkl (optional)


## Notes/Addendums
2. These scripts demonstrate methods available in TruEra Diagnostics, corresponding to release version 1.36.x.
1. The TruEra URL and access token are retrieved from shell environment vars. You can replace these lines in scripts 1, 2, and 4 with strings if preferred. Or, use the shell command export varname=value to create these variables for yourself in the shell you are executing these scripts from.


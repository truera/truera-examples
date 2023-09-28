#!/usr/bin/env python
# coding: utf-8

# # TruEra Python SDK
# ## Virtual Model Ingestion
# ## Sales Forecasting demo

import pandas as pd
import numpy as np
import pickle
import random

from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import TokenAuthentication
from truera.client.truera_authentication import BasicAuthentication
from truera.client.ingestion import ColumnSpec, ModelOutputContext

from truera.client.ingestion.util import merge_dataframes_and_create_column_spec

background_data_df=pd.read_csv("background_data_df.csv",index_col=[0])
lr_train_data_df=pd.read_csv("lr_train_data_df.csv",index_col=[0]) 
lr_val_data_df=pd.read_csv("lr_val_data_df.csv",index_col=[0])  
lr_prod_data_df=pd.read_csv("lr_prod_data_df.csv",index_col=[0]) 

rf_train_data_df=pd.read_csv("rf_train_data_df.csv",index_col=0)
rf_val_data_df=pd.read_csv("rf_val_data_df.csv",index_col=0)
rf_prod_data_df=pd.read_csv("rf_prod_data_df.csv",index_col=0)

##Column specs
## previously persisted from data merge script
## these can be generated using a helper function, if you don't already have them available
with open('column_spec.pkl', 'rb') as f:
    column_spec = pickle.load(f)

with open('prod_column_spec.pkl', 'rb') as f:
    prod_column_spec = pickle.load(f)

with open('background_column_spec.pkl', 'rb') as f:
    background_column_spec = pickle.load(f)
    
# Note that there are three separate column specs used:
# 1. background column spec -- index, pre, post, labels, and predictions (optional). No feature influences.
# 2. dev column spec -- index, pre, post, labels, predictions, and feature influences.
# 3. prod column spec -- index, pre, post, labels, predictions, feature influences, and timestamps. 

# connection details
TRUERA_URL = "https://app.truera.net"
AUTH_TOKEN = "<insert auth token>"

auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

import sys
project_name = str(sys.argv[1])
print(project_name)
tru.add_project(project_name, score_type='regression')
tru.activate_client_setting('create_model_tests_on_split_ingestion')

# ## Data Collection
# 1. Use data schema, pre- & post-feature engineering, to create feature map
# 2. Add new data collection to project with feature map

with open('feature_map.pkl', 'rb') as f:
    FEATURE_MAP = pickle.load(f)

tru.add_data_collection("OJ Sales Data LR", pre_to_post_feature_map=FEATURE_MAP, provide_transform_with_model=False)

# ## Model 1: Ridge Regression
# 1. add a 'virtual' model -- placeholder for associated I/O data that will be ingested
# 2. add data
# * background split - creates basis for interpretation of feature influences associated with subsequent dev & prod data
# * training data
# * validation data
# * production data

model_name = 'Ridge Regression'
tru.add_model(model_name)

# ### RF - Background data
#can use training data if you don't have separate/distinct background data you'd like to use
tru.add_data(
        data=background_data_df,
        data_split_name='background data',
        column_spec=background_column_spec,
        model_output_context=ModelOutputContext(
            model_name=model_name,
            influence_type='truera-qii',
            score_type='regression'))

# ### LR - Training Data
tru.add_data(
        data=lr_train_data_df,
        data_split_name='training data',
        column_spec=column_spec,
        model_output_context=ModelOutputContext(
            model_name=model_name,
            background_split_name='background data',
            influence_type='truera-qii',
            score_type='regression'))

# ### LR - Validation Data
tru.add_data(
        data=lr_val_data_df,
        data_split_name='validation data',
        column_spec=column_spec,
        model_output_context=ModelOutputContext(
            model_name=model_name,
            background_split_name='background data',
            influence_type='truera-qii',
            score_type='regression'))


# ### LR - Production Data
model_name = 'Ridge Regression'
tru.add_production_data(
        data=lr_prod_data_df,
        column_spec=prod_column_spec,
        model_output_context=ModelOutputContext(
            model_name=model_name,
            background_split_name='background data',
            influence_type='truera-qii',
            score_type='regression'))

# ## Model 2: Random Forest Regressor
# 1. add a 'virtual' model -- placeholder for associated I/O data that will be ingested
# 2. add data
# * background split - creates basis for interpretation of feature influences associated with subsequent dev & prod data
# * training data
# * validation data
# * production data

tru.add_data_collection("OJ Sales Data RF", pre_to_post_feature_map=FEATURE_MAP, provide_transform_with_model=False)

model_name = 'Random Forest Regressor'
tru.add_model(model_name)

### RF - Background data
tru.add_data(
        data=background_data_df,
        data_split_name='background data',
        column_spec=background_column_spec,
        model_output_context=ModelOutputContext(
            model_name=model_name,
            influence_type='truera-qii',
            score_type='regression'))

# ### RF - Training Data
tru.add_data(
        data=rf_train_data_df,
        data_split_name='training data',
        column_spec=column_spec,
        model_output_context=ModelOutputContext(
            model_name=model_name,
            background_split_name='background data',
            influence_type='truera-qii',
            score_type='regression'))

# ### RF - Validation Data
tru.add_data(
        data=rf_val_data_df,
        data_split_name='validation data',
        column_spec=column_spec,
        model_output_context=ModelOutputContext(
            model_name=model_name,
            background_split_name='background data',
            influence_type='truera-qii',
            score_type='regression'))

# ### RF - Production Data
tru.add_production_data(
        data=rf_prod_data_df,
        column_spec=prod_column_spec,
        model_output_context=ModelOutputContext(
            model_name=model_name,
            background_split_name='background data',
            influence_type='truera-qii',
            score_type='regression'))



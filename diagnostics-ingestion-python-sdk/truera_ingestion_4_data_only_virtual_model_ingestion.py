#!/usr/bin/env python
# coding: utf-8

# Script takes a single argument: a user-defined project name, as a string

# # TruEra Python SDK
# ## Virtual Model Ingestion
# ## Sales Forecasting demo

import pandas as pd
import numpy as np
import pickle
import random
import os
from datetime import date,datetime

from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import TokenAuthentication
from truera.client.ingestion import ColumnSpec, ModelOutputContext

background_data_df=pd.read_csv("./background_data_df.csv",index_col=[0])
rf_train_data_df=pd.read_csv("./rf_train_data_df.csv",index_col=0)

##Column specs
## previously persisted from data merge script
## these can be generated using a helper function, if you don't already have them available
with open('./column_spec.pkl', 'rb') as f:
    column_spec = pickle.load(f)

with open('./background_column_spec.pkl', 'rb') as f:
    background_column_spec = pickle.load(f)
    
# Note that there are three separate column specs used:
# 1. background column spec -- index, pre, post (required); labels, and predictions (optional). No feature influences.
# 2. dev column spec -- index, pre, post, labels, predictions, and feature influences.

AUTH_TOKEN = os.environ.get('AUTH_TOKEN')
TRUERA_URL = os.environ.get('URL')

# Python SDK - Create TruEra workspace
auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

import sys
project_name = "{} {}".format(str(sys.argv[1]), date.today())
print(project_name)

try:
    tru.add_project(project_name, score_type='regression')
except:
    tru.delete_project(project_name)
    tru.add_project(project_name, score_type='regression')
    
# ## Data Collection
# 1. Load existing feature map for pre-to-post feature mapping. For this use case, this was generated/saved as .pkl in script 1. 
# 2. Add new data collection to project with feature map

with open('./feature_map.pkl', 'rb') as f:
    FEATURE_MAP = pickle.load(f)

# ## Model: Random Forest Regressor
# 1. add a 'virtual' model -- placeholder for associated I/O data that will be ingested
# 2. add data
# * background split - creates basis for interpretation of feature influences associated with subsequent dev & prod data
# * training data
# * validation data -- placeholder
# * production data -- placeholder

tru.add_data_collection("OJ Sales Data", pre_to_post_feature_map=FEATURE_MAP, provide_transform_with_model=False)

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
# Add additional data splits using the same methods provided

# ### RF - Production Data
# Add additional data splits using the same methods provided



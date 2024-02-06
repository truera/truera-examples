#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import date, datetime

from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import TokenAuthentication

# connection details
AUTH_TOKEN = os.environ.get('AUTH_TOKEN')
TRUERA_URL = os.environ.get('URL')

# Python SDK - Create TruEra workspace
auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

project_name = "Sales Forecasting Scripted {}".format(date.today())
tru.set_project(project_name)

#predictions
tru.set_model("Random Forest Regressor")
tru.set_data_split("training data")
rf_train_preds = tru.get_ys_pred()

# ## Retrieve Feature Influences
tru.set_model("Random Forest Regressor")
tru.set_data_split("training data")
#note: here, we compute all feature influences, rather than just the first 1000, by using the stop parameter
rf_train_feat_infs = tru.get_feature_influences(stop=len(rf_train_preds))

#save feature influences
rf_train_feat_infs.to_csv('rf_train_FIs.csv')

#save predictions
rf_train_preds.to_csv('rf_train_preds.csv')

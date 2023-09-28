#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import random
import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import TokenAuthentication
from truera.client.truera_authentication import BasicAuthentication
from truera.client.ingestion import ColumnSpec, ModelOutputContext

# connection details
TRUERA_URL = "https://app.truera.net"
AUTH_TOKEN = "<insert auth token>"

auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

project_name = "Sales Forecasting"
tru.set_project(project_name)

# ## Retrieve Feature Influences
tru.set_model("Ridge Regression")
tru.set_data_split("training data")
lr_train_feat_infs = tru.get_feature_influences()
lr_train_feat_infs.to_csv('lr_train_FIs.csv')

tru.set_model("Ridge Regression")
tru.set_data_split("validation data")
lr_val_feat_infs = tru.get_feature_influences()
lr_val_feat_infs.to_csv('lr_val_FIs.csv')

tru.set_model("Random Forest Regressor")
tru.set_data_split("training data")
rf_train_feat_infs = tru.get_feature_influences()
rf_train_feat_infs.to_csv('rf_train_FIs.csv')

tru.set_model("Random Forest Regressor")
tru.set_data_split("validation data")
rf_val_feat_infs = tru.get_feature_influences()
rf_val_feat_infs.to_csv('rf_val_FIs.csv')

#predictions
tru.set_model("Ridge Regression")
tru.set_data_split("training data")
lr_train_preds = tru.get_ys_pred()
## Note: we need predictions, to generate feature influences. 
## In other words, (some) predictions are being generated as part of this call

tru.set_model("Ridge Regression")
tru.set_data_split("validation data")
lr_val_preds = tru.get_ys_pred()

tru.set_model("Random Forest Regressor")
tru.set_data_split("training data")
rf_train_preds = tru.get_ys_pred()

tru.set_model("Random Forest Regressor")
tru.set_data_split("validation data")
rf_val_preds = tru.get_ys_pred()

lr_train_preds.to_csv('lr_train_preds.csv')
lr_val_preds.to_csv('lr_val_preds.csv')
rf_train_preds.to_csv('rf_train_preds.csv')
rf_val_preds.to_csv('rf_val_preds.csv')

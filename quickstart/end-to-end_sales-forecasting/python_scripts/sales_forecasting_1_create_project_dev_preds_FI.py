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

from truera.client.ingestion.util import merge_dataframes_and_create_column_spec

# connection details
TRUERA_URL = "https://app.truera.net"
AUTH_TOKEN = "<insert auth token>"

auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

#create project
project_name = "Sales Forecasting"
tru.add_project(project_name, score_type="regression")
tru.activate_client_setting('create_model_tests_on_split_ingestion')

X_train_pre = pd.read_csv('./pre_train.csv',index_col=0).reset_index()
X_train_post = pd.read_csv('./post_train.csv',index_col=0).reset_index()
y = pd.read_csv('./labels_train.csv',index_col=0).reset_index()

FEATURE_MAP = {}
for post in X_train_post.drop(columns=['index','datetime']).columns:
    mapped = None
    for pre in X_train_pre.columns:
        if post.startswith(pre) and (mapped is None or len(mapped) < len(pre)):
            mapped = pre
    if mapped not in FEATURE_MAP:
        FEATURE_MAP[mapped] = []
    FEATURE_MAP[mapped].append(post)
    
#save feature map for future use
with open('feature_map.pkl', 'wb') as f:
    pickle.dump(FEATURE_MAP, f)

data_collection_name='OJ Sales Data'
tru.add_data_collection(data_collection_name, pre_to_post_feature_map=FEATURE_MAP, provide_transform_with_model=False)

data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data=X_train_pre,
                        post_data=X_train_post,
                        labels=y_df)

tru.add_data(
        data_split_name='training data',
        data=data_df,
        column_spec=column_spec)

X_val_pre = pd.read_csv('./split_sim/pre_split_2023-08-20.csv',index_col=0).reset_index()
X_val_post = pd.read_csv('./split_sim/post_split_2023-08-20.csv',index_col=0).reset_index()
y_val = pd.read_csv('./split_sim/label_2023-08-20.csv',index_col=0).reset_index()
#extra_val = pd.read_csv('./split_sim/extra_1.csv', index_col='datetime')

val_data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data=X_val_pre,
                        post_data=X_val_post,
                        labels=y_val)

tru.add_data(
        data_split_name='validation data',
        data=val_data_df,
        column_spec=column_spec)

lin_reg = pickle.load(open("linreg.pkl", 'rb'))
model_name = 'Ridge Regression'
tru.add_python_model(model_name, lin_reg)


# ## **Monitoring requirement: add new data collection for second model**
# - 1:1 dc:model req for monitoring
# - duplicated development data into 2nd data collection
#   - this is required to have access to FIs for all 6 model-dev_split combinations (see next section)
tru.add_data_collection("OJ Sales Data RF", pre_to_post_feature_map=FEATURE_MAP, provide_transform_with_model=False)

random_forest = pickle.load(open("rf.pkl", 'rb'))
model_name = 'Random Forest Regressor'
tru.add_python_model(model_name, random_forest)

data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data=X_train_pre,
                        post_data=X_train_post,
                        labels=y_df)

tru.add_data(
        data_split_name='training data',
        data=data_df,
        column_spec=column_spec)

tru.add_data(
        data_split_name='validation data',
        data=val_data_df,
        column_spec=column_spec)

# # Compute & Upload dev split Feature Influences using TruEra QII
    
tru.set_model("Ridge Regression")
tru.set_data_split("training data")
lr_train_feat_infs = tru.compute_feature_influences(stop=len(X_train_pre))
## Note: we need predictions, to generate feature influences. 
## In other words, (some) predictions are being generated as part of this call

tru.set_model("Ridge Regression")
tru.set_data_split("validation data")
lr_val_feat_infs = tru.compute_feature_influences()

tru.set_model("Random Forest Regressor")
tru.set_data_split("training data")
rf_train_feat_infs = tru.compute_feature_influences(stop=len(X_val_pre))

tru.set_model("Random Forest Regressor")
tru.set_data_split("validation data")
rf_val_feat_infs = tru.compute_feature_influences()

#this dc contains lin_reg / ridge regression model
tru.set_data_collection("OJ Sales Data")

preds = lin_reg.predict(X_train_post.drop(columns=['datetime','index']))
preds_df = pd.DataFrame(preds, columns = ['logmove'], index=[X_train_post['index'], X_train_post.datetime])
lr_train_preds= preds_df.reset_index()

preds = lin_reg.predict(X_val_post.drop(columns=['datetime','index']))
preds_df = pd.DataFrame(preds, columns = ['logmove'], index=[X_val_post['index'], X_val_post.datetime])
lr_val_preds = preds_df.reset_index() #index as column

#this DC contains Random Forest Regressor / RF model
tru.set_data_collection("OJ Sales Data RF")
preds = random_forest.predict(X_train_post.drop(columns=['index','datetime']))
preds_df = pd.DataFrame(preds, columns = ['logmove'], index=[X_train_post['index'],X_train_post.datetime])
rf_train_preds = preds_df.reset_index()

preds = random_forest.predict(X_val_post.drop(columns=['index','datetime']))
preds_df = pd.DataFrame(preds, columns = ['logmove'], index=[X_val_post['index'], X_val_post.datetime])
rf_val_preds = preds_df.reset_index()

tru.add_data(
    data=lr_train_preds,
    data_split_name="training data",
    column_spec=ColumnSpec(
        id_col_name="index",
        timestamp_col_name='datetime',
        prediction_col_names='logmove'),
        
    model_output_context=ModelOutputContext(
        model_name="Ridge Regression",
        score_type='regression')
    )

tru.add_data(
    data=lr_val_preds,
    data_split_name="validation data",
    column_spec=ColumnSpec(
        id_col_name="index",
        timestamp_col_name='datetime',
        prediction_col_names='logmove'
    ),
    model_output_context=ModelOutputContext(
        model_name="Ridge Regression",
        score_type='regression')
)

tru.add_data(
    data=rf_train_preds,
    data_split_name="training data",
    column_spec=ColumnSpec(
        id_col_name="index",
        timestamp_col_name='datetime',
        prediction_col_names='logmove'
    ),
    model_output_context=ModelOutputContext(
        model_name="Random Forest Regressor",
        score_type='regression')
)

tru.add_data(
    data=rf_val_preds,
    data_split_name="validation data",
    column_spec=ColumnSpec(
        id_col_name="index",
        timestamp_col_name='datetime',
        prediction_col_names='logmove'
    ),
    model_output_context=ModelOutputContext(
        model_name="Random Forest Regressor",
        score_type='regression')
)


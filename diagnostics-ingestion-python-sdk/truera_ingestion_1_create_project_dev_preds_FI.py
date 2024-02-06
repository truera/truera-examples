#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import random
from random import randrange
from datetime import date, datetime

import sklearn
from sklearn.ensemble import RandomForestClassifier

from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import TokenAuthentication
from truera.client.truera_authentication import BasicAuthentication
from truera.client.ingestion import ColumnSpec, ModelOutputContext

from truera.client.ingestion.util import merge_dataframes_and_create_column_spec

# connection details
AUTH_TOKEN = os.environ.get('AUTH_TOKEN')
TRUERA_URL = os.environ.get('URL')

# Python SDK - Create TruEra workspace
auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

#note: use truera-qii if possible/truera-qii installed. Otherwise, omit this setting; TruEra will use the OSS SHAP library that corresponds to your model and prediction type. Be aware that this may lead to lengthy increases in computation time to generate Shapley value estimates.
tru.set_influence_type('truera-qii')

#create project
version=randrange(1000)

project_name = "Sales Forecasting Scripted {}".format(date.today())
#project_name = "Sales Forecasting"
try:
    tru.add_project(project_name, score_type="regression")
except:
    tru.delete_project(project_name)
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

tru.add_data_collection("OJ Sales Data", pre_to_post_feature_map=FEATURE_MAP, provide_transform_with_model=False)

random_forest = pickle.load(open("rf_v1.pkl", 'rb'))
model_name = 'Random Forest Regressor'
tru.add_python_model(model_name, random_forest)

data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        extra_data=X_train_pre[['index','datetime']],
                        pre_data=X_train_pre.drop(columns=['datetime']),
                        post_data=X_train_post.drop(columns=['datetime']),
                        labels=y)

tru.add_data(
        data_split_name='training data',
        data=data_df,
        column_spec=column_spec)

# # Computing & adding Feature Influences
# # Compute & Feature Influences using TruEra QII, for existing model/data
tru.set_model("Random Forest Regressor")
tru.set_data_split("training data")
infs_df=tru.get_feature_influences().reset_index()

#influences - add to project
tru.add_model_feature_influences(feature_influence_data=infs_df, id_col_name='index')

# # Computing & adding Predictions
#this DC contains Random Forest Regressor / RF model
tru.set_model("Random Forest Regressor")
tru.set_data_split("training data")

#note: could also use tru.get_ys_preds() here, in similar fashion to tru.get_feature_influences(), above
preds = random_forest.predict(X_train_post.drop(columns=['index','datetime']))
preds_df = pd.DataFrame(preds, columns = ['logmove'], index=[X_train_post['index'],X_train_post.datetime])
rf_train_preds = preds_df.reset_index()

# Add predictions to project. Also can use convenience function .add_model_predictions that assumes model output context from existing truera workspace context. Here, we use .add_data to demonstrate 'stateless' use. 

tru.add_data(
    data=rf_train_preds,
    data_split_name="training data",
    column_spec=ColumnSpec(
        id_col_name="index",
        prediction_col_names='logmove'
    ),
    model_output_context=ModelOutputContext(
        model_name="Random Forest Regressor",
        score_type='regression')
)

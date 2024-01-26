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

import glob
import os
from datetime import datetime

from truera.client.ingestion.util import merge_dataframes_and_create_column_spec

# connection details
TRUERA_URL = "https://app.truera.net"
AUTH_TOKEN = "<insert auth token>"

auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

project_name = "Sales Forecasting"
tru.set_project(project_name)

def load_prod_data(start, end):
    start = datetime.strptime(start, '%Y-%m-%d').date()
    end = datetime.strptime(end, '%Y-%m-%d').date()
    print(type(start))
    
    #gather files to include
    f_prod = glob.glob(os.path.join('./split_sim', 'pre_split_*.csv'))
    f_prod_post= glob.glob(os.path.join('./split_sim', "post_split_*.csv"))
    f_y_prod = glob.glob(os.path.join('./split_sim', "label_*.csv"))
    
    #sort file names
    f_prod.sort()
    f_prod_post.sort()
    f_y_prod.sort()
    
    X_prod = pd.concat((pd.read_csv(f,index_col=0).reset_index() for f in f_prod), ignore_index=True)
    X_prod_post = pd.concat((pd.read_csv(f,index_col=0).reset_index() for f in f_prod_post), ignore_index=True)
    y_prod = pd.concat((pd.read_csv(f,index_col=0).reset_index() for f in f_y_prod), ignore_index=True)
        
    prod_data_df, column_spec = merge_dataframes_and_create_column_spec(id_col_name='index',
                                                                        timestamp_col_name='datetime',
                                                                        pre_data=X_prod,
                                                                        post_data=X_prod_post,
                                                                        labels=y_prod)
    #greater than the start date and smaller than the end date
    prod_data_df['datetime'] = pd.to_datetime(prod_data_df['datetime']).dt.date
    prod_data_df = prod_data_df[(prod_data_df['datetime'] >= start) & (prod_data_df['datetime'] <= end)]
    print(prod_data_df.datetime.min())
    print(prod_data_df.datetime.max())
    print(prod_data_df.shape)
    print(column_spec)
    return prod_data_df, column_spec

prod_data_df, prod_column_spec = load_prod_data('2023-08-24', '2023-09-20')
prod_data_df.to_csv('prod_df.csv',index=False)

with open('prod_column_spec.pkl', 'wb') as f:
    pickle.dump(prod_column_spec, f)
    
def generate_prod_preds(model, data):
    preds = model.predict(data.drop(columns=data.columns.difference(column_spec.post_data_col_names)))
    preds_df = pd.DataFrame(preds, columns = ['preds'], index=[data['index'], data.datetime])
    preds_df = preds_df.reset_index()
    print(preds_df.shape)

    return preds_df

lin_reg = pickle.load(open("linreg.pkl", 'rb'))
random_forest = pickle.load(open("rf.pkl", 'rb'))

lr_prod_preds = generate_prod_preds(lin_reg, prod_data_df)
rf_prod_preds = generate_prod_preds(random_forest, prod_data_df)

lr_prod_preds.to_csv('lr_prod_preds.csv',index=False)
rf_prod_preds.to_csv('rf_prod_preds.csv',index=False)

#FIs
tru.set_data_collection('OJ Sales Data')
tru.set_model('Ridge Regression')
LR_explainer = tru.get_explainer()

LR_prod_FIs = LR_explainer.compute_feature_influences_for_data(
    pre_data = prod_data_df[column_spec.pre_data_col_names], 
    post_data = prod_data_df[column_spec.post_data_col_names], 
    ys = prod_data_df[column_spec.label_col_names])

LR_prod_FIs['index'] = prod_data_df[column_spec.id_col_name]

tru.set_data_collection("OJ Sales Data RF")
tru.set_model("Random Forest Regressor")
RF_explainer = tru.get_explainer()

RF_prod_FIs = RF_explainer.compute_feature_influences_for_data(
    pre_data = prod_data_df[column_spec.pre_data_col_names],
    post_data = prod_data_df[column_spec.post_data_col_names],
    ys = prod_data_df[column_spec.label_col_names])

RF_prod_FIs['index'] = prod_data_df[column_spec.id_col_name]

#persist for virtual ingestion / other uses
LR_prod_FIs.to_csv('lr_prod_FIs.csv',index=True)
RF_prod_FIs.to_csv('rf_prod_FIs.csv',index=True)
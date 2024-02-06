#!/usr/bin/env python
# coding: utf-8

# # Merge Dataframes to prepare for virtual model ingestion
# - input data
# - predictions
# - feature influences
# 
# merge and create column specs, for virtual model ingestion

import pandas as pd
import numpy as np
import pickle

#note that we just use these functions as utilities here -- we don't actually connect to a TruEra workspace in this script
from truera.client.ingestion import ColumnSpec, ModelOutputContext
from truera.client.ingestion.util import merge_dataframes_and_create_column_spec

X_train_pre = pd.read_csv('./pre_train.csv',index_col=0).reset_index()
X_train_post = pd.read_csv('./post_train.csv',index_col=0).reset_index()
y = pd.read_csv('./labels_train.csv',index_col=0).reset_index()

# ## Load pre-computed predictions
rf_train_preds = pd.read_csv('rf_train_preds.csv',index_col=0).reset_index()
rf_train_preds=rf_train_preds.rename(columns={rf_train_preds.columns[0]: 'index'})

# ## Load pre-computed feature influences
rf_train_FIs = pd.read_csv('rf_train_FIs.csv',index_col=0).reset_index()
rf_train_FIs=rf_train_FIs.rename(columns={rf_train_FIs.columns[0]: 'index'})

#merge data
print(X_train_pre.shape)
print(X_train_post.shape)
print(y.shape)
print(rf_train_preds.shape)
print(rf_train_FIs.shape)

rf_train_data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        pre_data=X_train_pre.drop(columns=['datetime']),
                        post_data=X_train_post.drop(columns=['datetime']),
                        labels=y,
                        extra_data=X_train_pre[['index','datetime']],
                        predictions=rf_train_preds,                        
                        feature_influences=rf_train_FIs)

#save column spec as pickle file, for future use
with open('column_spec.pkl', 'wb') as f:
    pickle.dump(column_spec, f)

#save merged dataframe for future use
rf_train_data_df.to_csv("rf_train_data_df.csv") 

#save copy of training data as 'background data'
#note that we do not include predictions, feature influences, or labels in the background split. All we need is inputs. 
background_data_df, background_column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        pre_data=X_train_pre.drop(columns=['datetime']),
                        post_data=X_train_post.drop(columns=['datetime']),
                        labels=y,
                        extra_data=X_train_pre[['index','datetime']],
                        predictions=rf_train_preds)

print(background_column_spec)

background_data_df.to_csv("background_data_df.csv")

with open('background_column_spec.pkl', 'wb') as f:
    pickle.dump(background_column_spec, f)
        


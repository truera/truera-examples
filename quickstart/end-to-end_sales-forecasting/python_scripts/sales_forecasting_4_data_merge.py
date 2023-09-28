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

from truera.client.ingestion import ColumnSpec, ModelOutputContext
from truera.client.ingestion.util import merge_dataframes_and_create_column_spec
from truera.client.ingestion.util import merge_dataframes_and_create_column_spec

X_train_pre = pd.read_csv('./pre_train.csv',index_col=0).reset_index()
X_train_post = pd.read_csv('./post_train.csv',index_col=0).reset_index()
y = pd.read_csv('./labels_train.csv',index_col=0).reset_index()

X_val_pre = pd.read_csv('./split_sim/pre_split_2023-08-20.csv',index_col=0).reset_index()
X_val_post = pd.read_csv('./split_sim/post_split_2023-08-20.csv',index_col=0).reset_index()
y_val = pd.read_csv('./split_sim/label_2023-08-20.csv',index_col=0).reset_index()

prod_data_df=pd.read_csv('prod_df.csv',index_col=0).reset_index()

# ## Predictions
lr_train_preds = pd.read_csv('lr_train_preds.csv',index_col=0).reset_index()
lr_train_preds=lr_train_preds.rename(columns={lr_train_preds.columns[0]: 'index'})

lr_val_preds = pd.read_csv('lr_val_preds.csv',index_col=0).reset_index()
lr_val_preds=lr_val_preds.rename(columns={lr_val_preds.columns[0]: 'index'})

rf_train_preds = pd.read_csv('rf_train_preds.csv',index_col=0).reset_index()
rf_train_preds=rf_train_preds.rename(columns={rf_train_preds.columns[0]: 'index'})
rf_val_preds = pd.read_csv('rf_val_preds.csv',index_col=0).reset_index()
rf_val_preds=rf_val_preds.rename(columns={rf_val_preds.columns[0]: 'index'})

lr_prod_preds = pd.read_csv('lr_prod_preds.csv',index_col=0).reset_index()
rf_prod_preds = pd.read_csv('rf_prod_preds.csv',index_col=0).reset_index()

# ## Feature Influences
lr_train_FIs = pd.read_csv('lr_train_FIs.csv',index_col=0).reset_index()
lr_train_FIs=lr_train_FIs.rename(columns={lr_train_FIs.columns[0]: 'index'})
lr_val_FIs = pd.read_csv('lr_val_FIs.csv',index_col=0).reset_index()
lr_val_FIs=lr_val_FIs.rename(columns={lr_val_FIs.columns[0]: 'index'})

rf_train_FIs = pd.read_csv('rf_train_FIs.csv',index_col=0).reset_index()
rf_train_FIs=rf_train_FIs.rename(columns={rf_train_FIs.columns[0]: 'index'})
rf_val_FIs = pd.read_csv('rf_val_FIs.csv',index_col=0).reset_index()
rf_val_FIs=rf_val_FIs.rename(columns={rf_val_FIs.columns[0]: 'index'})

rf_prod_FIs = pd.read_csv('rf_prod_FIs.csv',index_col=0)
lr_prod_FIs = pd.read_csv('lr_prod_FIs.csv',index_col=0)

#dev data
lr_train_data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data=X_train_pre,
                        post_data=X_train_post,
                        labels=y,
                        predictions=lr_train_preds,
                        feature_influences=lr_train_FIs)

lr_val_data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data=X_val_pre,
                        post_data=X_val_post,
                        labels=y_val,
                        predictions=lr_val_preds,
                        feature_influences=lr_val_FIs)

rf_train_data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data=X_train_pre,
                        post_data=X_train_post,
                        labels=y,
                        predictions=rf_train_preds,                        
                        feature_influences=rf_train_FIs)

rf_val_data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data=X_val_pre,
                        post_data=X_val_post,
                        labels=y_val,
                        predictions=rf_val_preds,
                        feature_influences=rf_val_FIs)

with open('column_spec.pkl', 'wb') as f:
    pickle.dump(column_spec, f)

#production data
lr_prod_data_df, prod_column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data = prod_data_df[column_spec.pre_data_col_names+[column_spec.id_col_name]], 
                        post_data = prod_data_df[column_spec.post_data_col_names+[column_spec.id_col_name]], 
                        labels = prod_data_df[column_spec.label_col_names+[column_spec.id_col_name]],
                        predictions = lr_prod_preds,
                        feature_influences = lr_prod_FIs)



rf_prod_data_df, prod_column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data = prod_data_df[column_spec.pre_data_col_names+[column_spec.id_col_name]], 
                        post_data = prod_data_df[column_spec.post_data_col_names+[column_spec.id_col_name]], 
                        labels = prod_data_df[column_spec.label_col_names+[column_spec.id_col_name]],
                        predictions = rf_prod_preds,
                        feature_influences = rf_prod_FIs)
print(prod_column_spec)

with open('prod_column_spec.pkl', 'wb') as f:
    pickle.dump(prod_column_spec, f)

lr_train_data_df.to_csv("lr_train_data_df.csv") 
lr_val_data_df.to_csv("lr_val_data_df.csv")  
lr_prod_data_df.to_csv("lr_prod_data_df.csv") 

rf_train_data_df.to_csv("rf_train_data_df.csv") 
rf_val_data_df.to_csv("rf_val_data_df.csv")
rf_prod_data_df.to_csv("rf_prod_data_df.csv") 

background_data_df, background_column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime',
                        pre_data=X_train_pre,
                        post_data=X_train_post)
print(background_column_spec)

background_data_df.to_csv("background_data_df.csv")

with open('background_column_spec.pkl', 'wb') as f:
    pickle.dump(background_column_spec, f)
        


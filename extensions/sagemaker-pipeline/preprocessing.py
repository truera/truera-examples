import argparse
import os
import requests
import tempfile
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sys, os, subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#install("numpy>=1.21")
install("truera")
#install("numpy==1.24.1")
#install("protobuf==3.20.2")

from truera.client.truera_workspace import TrueraWorkspace
#from truera.client.truera_authentication import TokenAuthentication
#from truera.client.truera_authentication import BasicAuthentication
from truera.client.truera_authentication import ServiceAccountAuthentication
from truera.client.ingestion import ColumnSpec, ModelOutputContext

# Because this is a headerless CSV file, specify the column names here.
feature_columns_names = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
label_column = "rings"

feature_columns_dtype = {
    "sex": str,
    "length": np.float64,
    "diameter": np.float64,
    "height": np.float64,
    "whole_weight": np.float64,
    "shucked_weight": np.float64,
    "viscera_weight": np.float64,
    "shell_weight": np.float64
}
label_column_dtype = {"rings": np.float64}


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

##Pre-processing + TruEra ingestion
if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/abalone-dataset.csv",
        header=None, 
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)
    )
    numeric_features = list(feature_columns_names)
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    y = df.pop("rings")
    X_pre = preprocess.fit_transform(df)

    #get post transform column names, as list
    ohe_feat = preprocess.transformers_[1][1]\
        ['onehot'].get_feature_names_out()
    cat_post_list = ohe_feat.tolist()
    input_columns = ['rings']+numeric_features+cat_post_list

    y_pre = y.to_numpy().reshape(len(y), 1)
    X = np.concatenate((y_pre, X_pre), axis=1)

    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])

    train_df = pd.DataFrame(train)
    train_df.columns = [str(c) for c in input_columns]
    train_df.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation_df = pd.DataFrame(validation)
    validation_df.columns = [str(c) for c in input_columns]
    validation_df.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test_df = pd.DataFrame(test)
    test_df.columns = [str(c) for c in input_columns]
    test_df.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    ##########################
    ## Truera code ##
    ##########################
    
    auth = ServiceAccountAuthentication(
    client_id = os.environ['TRUERA_CLIENT_ID'],
    client_secret = os.environ['TRUERA_SECRET'])
    
    tru = TrueraWorkspace(os.environ['TRUERA_CONNECTION_STRING'], auth,verify_cert=False)
    tru.set_project(os.environ['TRUERA_PROJECT_NAME'])
    tru.set_data_collection(os.environ['TRUERA_DATA_COLLECTION_NAME'])
    
    ids = 'index'
    
    #we are not using a feature map here, so model-readable columns are read in as "pre" data
    print(train_df.shape)
    tru.add_data(data=train_df.reset_index(), 
                 data_split_name='train',
                 column_spec=ColumnSpec(id_col_name=ids,
                                        pre_data_col_names=input_columns[1:],
                                        label_col_names=input_columns[0]))
    print(validation_df.shape)
    tru.add_data(data=validation_df.reset_index(), 
                 data_split_name='validation',
                 column_spec=ColumnSpec(id_col_name=ids,
                                        pre_data_col_names=input_columns[1:],
                                        label_col_names=input_columns[0]))

    print(test_df.shape)
    tru.add_data(data=test_df.reset_index(), 
                 data_split_name='test',
                 column_spec=ColumnSpec(id_col_name=ids,
                                        pre_data_col_names=input_columns[1:],
                                        label_col_names=input_columns[0]))


import sys, os, subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("numpy>=1.21")
install("truera")
#install("numpy==1.24.1")
#install("protobuf==3.20.2")

#import shap
from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import TokenAuthentication
from truera.client.truera_authentication import BasicAuthentication
from truera.client.truera_authentication import ServiceAccountAuthentication
from truera.client.ingestion import ColumnSpec, ModelOutputContext

import json
import pathlib
import tarfile
import numpy as np
import pandas as pd
import xgboost

import os

from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    cmd = 'ls -lar'
    os.system(cmd)
    model = xgboost.Booster()
    model.load_model('xgboost-model')

    ##########################
    ## Truera code ##
    ##########################

    auth = ServiceAccountAuthentication(
    client_id = os.environ['TRUERA_CLIENT_ID'],
    client_secret = os.environ['TRUERA_SECRET'])
    
    tru = TrueraWorkspace(os.environ['TRUERA_CONNECTION_STRING'], auth, verify_cert=False)
    tru.set_project(os.environ['TRUERA_PROJECT_NAME'])
    tru.set_data_collection(os.environ['TRUERA_DATA_COLLECTION_NAME'])
    tru.set_model_execution("local")
    
    #add model object
    tru.add_python_model("xgb_abalone_regression", model)
    
    for split in tru.get_data_splits():
        print('loading data split {}'.format(split))
        tru.set_data_split(split)
        temp = tru.get_xs()
        DMtemp = xgboost.DMatrix(temp)
        
        print('generating predictions for split {}'.format(split))
        preds = model.predict(DMtemp)
        preds_df = pd.DataFrame(preds, index=temp.index, columns=['rings']).reset_index()

        print('adding predictions to data split {}'.format(split))
        tru.add_data(
        data=preds_df,
        data_split_name=split,
        column_spec=ColumnSpec(
            id_col_name="index",
            prediction_col_names='rings'),

        model_output_context=ModelOutputContext(
            model_name=tru.get_models()[0],
            score_type='regression')
        )

        print('generating and uploading feature influences for split {}'.format(split))
        tru.compute_feature_influences()
        print('generating and uploading error influences for split {}'.format(split))
        tru.compute_error_influences()

    #Example Tests
    tru.tester.add_performance_test(test_name="MSE test",
                                    data_split_names=tru.get_data_splits(), 
                                    metric="MSE", 
                                    warn_if_greater_than = 3,
                                    fail_if_greater_than = 6)

    tru.tester.add_performance_test(test_name="MAPE test",
                                    data_split_names=tru.get_data_splits(), 
                                    metric="MAPE", 
                                    warn_if_greater_than = 8,
                                    fail_if_greater_than = 16)

    tru.tester.add_performance_test(test_name="MAE test",
                                    data_split_names=tru.get_data_splits(), 
                                    metric="MAE", 
                                    warn_if_greater_than = 2,
                                    fail_if_greater_than = 4)

    metric_dict = tru.tester.get_model_test_results().as_dict()["Performance Tests"]
    metric_df = pd.DataFrame(metric_dict["Rows"], columns=metric_dict["Column Names"])

    report_dict = {
        "regression_metrics": {}
    }

    num_tests = 0
    num_passed = 0
    for metric in metric_df["Metric"].unique():
        num_tests += 1
        row = metric_df[(metric_df["Metric"] == metric) & (metric_df["Split"] == "test")].iloc[0]
        score = row['Score']
        print(f"Metric: {metric} \t Value: {score} \t Outcome: {row['Outcome']}")

        if row["Outcome"] == "PASSED":
            num_passed += 1

        report_dict["regression_metrics"][metric] = {"value": score}

    report_dict["test_metrics"] = {
        "total_tests": num_tests,
        "num_passing": num_passed,
        "passing_percentage": num_passed / num_tests,
    }
    report_dict["test_results"] = metric_dict
    print(f"Tests passing: {num_passed}/{num_tests}")

    ##############################
    ## END Truera-specific code ##
    ##############################

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

# TruSHAP

### What's this useful for?
TruSHAP is an easy introduction to TruEra designed with SHAP users in mind. 

Add your model to a TruEra deployment with only 2! code changes to your notebooks that already use SHAP.

### All you need to do to unlock TruEra capabilities is:
1. Change the import from "import shap" to "import trushap as shap"
2. Add CONNECTION_STRING, and TOKEN as arguments to shap.Explainer()

Once you've made those two changes, the model will be added to a new project and data collection in your TruEra deployment.

When you use the explainer, e.g. explainer(X), the data split X will also be added to your TruEra deployment.

```python
# import trushap, and optionally alias as shap to preserve your SHAP code.
from truera.client.experimental.trushap import trushap as shap

# Initialize the explainer. 
# Include TruEra authentication (optional) to add the model to your TruEra deployment.
explainer = shap.Explainer(model, connection_string = CONNECTION_STRING, token = TOKEN)

# Calculate shapley values AND add data split to your TruEra deployment.
shap_values = explainer(X)
```

* This SHAP Extension allows you to get familiar with TruEra ingestion and Trustworthy ML concepts while using the model-agnostic SHAP interface. Creating the explainer in TruSHAP automatically ingests your model and data to your TruEra application without any additional lines of code.

* Once you get up and running with TruSHAP, transitioning to the TruEra way of analyzing models comes easy.

* All you have to do is add your TruEra connection details, the TruSHAP API does the rest.

### And *all* your existing SHAP code still works.

```python
shap.summary_plot(shap_values, X)
```

### Rather than using the automatically generated names, say you want to add your own names. All you have to do is add these as arguments. Then you can easily call them later when you want to test, optimize or compare it with other models.

```python
explainer = shap.Explainer(model,
                            connection_string = CONNECTION_STRING,
                            token = TOKEN,
                            project = "Adult Census Example",
                            data_collection_name = "Adult Census Data Collection",
                            model_name = "XGBRegressor V1"
                            )

shap_values = explainer(X, data_split_name = "all", split_type = "all")
```

### And don't forget the label data! Add labels to use with both SHAP and TruEra using the same argument (y).

```python
shap_values = explainer(X, data_split_name = "all_w_labels", y = y, split_type = "all")
```

### Adding multiple models for comparison is easy. Even specify metadata for each model such as the training split and training parameters.

```python
#Add multiple models
xgb_model = xgboost.XGBRegressor(max_depth = 6, min_child_weight = 2).fit(X_train, y_train)

explainer_xgb = shap.Explainer(xgb_model,
                            connection_string = CONNECTION_STRING,
                            token = TOKEN,
                            project = "Adult Census Model Comparison",
                            data_collection_name = "Adult Census Data Collection",
                            model_name = "XGBRegressor",
                            train_split = "train",
                            train_parameters = {"max_depth":6,
                                                "min_child_weight":2}
                            )

shap_values_xgb_train = explainer_xgb(X_train, data_split_name = "train", y = y_train, split_type = "train")
shap_values_xgb_test = explainer_xgb(X_test, data_split_name = "test", y = y_test, split_type = "test")

tree_model = DecisionTreeRegressor(max_depth=6).fit(X_train, y_train)

explainer_tree = shap.Explainer(tree_model,
                            connection_string = CONNECTION_STRING,
                            token = TOKEN,
                            project = "Adult Census Model Comparison",
                            data_collection_name = "Adult Census Data Collection",
                            model_name = "Decision Tree Regression",
                            train_split = "train",
                            train_parameters = {"max_depth": 6}
                            )

shap_values_tree_train = explainer_tree(X_train, data_split_name = "train", y = y_train, split_type = "train")
shap_values_tree_test = explainer_tree(X_test, data_split_name = "test", y = y_test, split_type = "test")
```

### For even more cool stuff, fetch the TruEra workspace to:
* Get feature importances
* Plot influence sensitivity plots (ISPs)

```python
tru = explainer.get_truera_workspace()

tru_explainer = tru.get_explainer('all_w_labels')

tru_explainer.get_global_feature_importances()

tru_explainer.plot_isp('Age')
```

### And find hotspots where the model underperforms:

```python
tru_explainer.suggest_high_error_segments()
```

### And once you have the TruEra workspace, get serious with the Test Harness to evaluate performance, fairness, feature importance and stability tests on your models.

1. Performance tests warn and/or fail if any of a number of metrics (accuracy, precision, AUC, etc) reaches a specified threshold.

2. Fairness tests establish criteria to compare a protected segment against the rest of the population.

3. Feature Importance tests ensure there are not too many unimportant features in the model.

4. Stability ensures that the behavior of the model is similar across two distributions.

```python
#set environment to the remote project, set context in remote
tru.set_environment("remote")
tru.set_project("Adult Census Model Comparison")
tru.set_data_collection("Adult Census Data Collection")
tru.set_data_split("train")

#set up protected segment for fairness test
tru.add_segment_group(name = "Gender", segment_definitions = dict({"Male": 'Sex == 1', 'Female': 'Sex == 0'}) )
tru.set_as_protected_segment(segment_group_name = "Gender", segment_name = "Female")

#performance test
for split_name in ["train", "test"]:
    tru.tester.add_performance_test(
        test_name = "Performance Test 1",
        data_split_names = [split_name],
        fail_if_greater_than = 0.3,
        metric = "RMSE")

    #fairness test
    tru.tester.add_fairness_test(data_split_name = split_name,
    segment_group_name = "Gender",
    protected_segment_name = "Female",
    metric = "MEAN_SCORE_DIFFERENCE",
    fail_if_outside = [-0.1,0.1], warn_if_outside = [-0.05, 0.05])

    #feature importance test
    tru.tester.add_feature_importance_test(data_split_name = split_name,
                                            min_importance_value= 0.02,
                                            warn_if_greater_than = 0,
                                            fail_if_greater_than = 5,
                                            overwrite = True)

#stability test
tru.tester.add_stability_test(base_data_split_name="train",
                                comparison_data_split_names=["test"],
                                metric="DIFFERENCE_OF_MEAN",
                                warn_if_greater_than = 0.1)
```

### View the Model Leaderboard
```python
tru.tester.get_model_leaderboard()
```

### And the Model Summary

```python
tru.set_model("XGBRegressor")
tru.tester.get_model_test_results()
```

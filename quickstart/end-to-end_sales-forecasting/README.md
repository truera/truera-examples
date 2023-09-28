# End-to-end regression modeling & TruEra ingestion demo

### Contents: End to end notebook + reference scripts for a set of semi-overlapping ingestion scenarios, based on OJ sales forecasting dataset. 

The scripts have numeric designations corresponding to the following contents
1. create new project, ingest 2 models and 2 data splits, with feature influences and predictions
2. reference an existing project to recall/persist existing predictions and feature influences
3. use an existing diagnostics project to (1) add production data and (2) generate predictions & feature influences for that production data
4. merge inputs, outputs, and feature influences for any given ‘data split’ (including production data for a specific timeframe) into a single ‘table’ of model I/Os
5. full virtual model ingestion based on (3), (4)

### Additional notes
* The scripts contain simple code snippets for pickling and persisting feature maps, and column_specs, for use across pipeline steps

* These come in very handy when loading new data that corresponds to an existing model/data schema

## More info in dataset and notebook contents:
Sales Forecasting models based on:
- https://docs.microsoft.com/en-us/azure/open-datasets/dataset-oj-sales-simulated?tabs=azureml-opendatasets#columns

Data is available from:
- http://www.cs.unitn.it/~taufer/Data/oj.csv

Data Dictionary and analysis exists here: 
- http://www.cs.unitn.it/~taufer/QMMA/L10-OJ-Data.html

Resources:
1. The notebook 'sales_forecasting.ipynb' contains code for:
  a. data loading and basic edav
  b. preparing data for modeling
  c. creating simulated partitions for future monitoring purposes
  d. training two models: a ridge regression (RR) model, and a random forest regressor (RFR)
  e. creating a truera project, adding data, adding the models
  
Notes:
  - little attempt made to tune models
  - RFR has better fit on training data, but does very poorly on test split (worse than RR)
  - Demographic information was trained upon 
  - Target is the log of unit sales
  - Arbitrary 'week' timestamps were converted to 2023 dates. 

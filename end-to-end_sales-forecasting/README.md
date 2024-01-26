# End-to-end regression modeling & TruEra ingestion demo

## Contents: 
* End to end notebook
* Reference scripts for a set of semi-overlapping ingestion scenarios, based on OJ sales forecasting dataset. 

### Python scripts 
The scripts have numeric designations corresponding to the following contents
1. create new project, ingest 2 models and 2 data splits; generate and ingest feature influences and predictions
2. reference an existing project to recall/persist existing predictions and feature influences
3. use an existing diagnostics project to (1) add production data and (2) generate predictions & feature influences for that production data
4. "Virtual Model" data prep - demonstration of data organization for 'data only' project setup, using I/Os from model object rather than the object itself
  - merges inputs, outputs, and feature influences for any given ‘data split’, including production data for a specific timeframe) into a single ‘table’ of model I/Os
5. Ingestion demonstration: 'virtual model', aka data only. 
  - TruEra project setup/creation, based on (3), (4)
  - uses previously pickled/saved feature map and column specs. For the purposes of adapting these utilities to your purposes, these can be created manually if not previously saved.

#### Script notes: 
* The scripts contain simple code snippets for pickling and persisting feature maps, and column_specs, for use across pipeline steps
* These come in very handy when loading new data that corresponds to an existing model/data schema

6. The notebook 'sales_forecasting.ipynb' contains code for:
  a. data loading and basic exploratory data analysis & visualization (EDAV)
  b. preparing data for modeling
  c. creating simulated partitions for future monitoring purposes
  d. training two models: a ridge regression (RR) model, and a random forest regressor (RFR)
  e. creating a truera project, adding data, adding the models
  
## Notes:
  - little attempt made to tune models
  - Random Forest Regressor (RFR) has better fit on training data, but does very poorly on test split, i.e., worse than Ridge Regression (RR)
  - Demographic information was trained upon -- consider whether this is appropriate; set up fairness tests if intersted in further exploration
  - Target is the log of unit sales
  - Arbitrary 'week' timestamps were converted to 2023 dates, for monitoring simulation purposes. 

## More info in dataset and notebook contents:
Sales Forecasting models based on:
- https://docs.microsoft.com/en-us/azure/open-datasets/dataset-oj-sales-simulated?tabs=azureml-opendatasets#columns

### Data is available from:
- http://www.cs.unitn.it/~taufer/Data/oj.csv

### Data Dictionary and analysis exists here: 
- http://www.cs.unitn.it/~taufer/QMMA/L10-OJ-Data.html
  

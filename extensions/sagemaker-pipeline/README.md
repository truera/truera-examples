# Sagemaker pipeline integration

These resources provide a user a template for injecting TruEra into an existing Sagemaker pipeline.

## There are two key steps in the pipeline that use TruEra's code:
1. preprocessing
2. evaluation

### Preprocessing:
- loads prepared modeling data into a TruEra project

### Evaluation:
- generates predictions and feature influences for data loaded in during the pre-processing step, using the model generated during the training step in the sagemaker pipeline.

##Resources:
- "Local" notebook showing how this flow works outside of Sagemaker
- "Sagemaker Pipeline" notebook generates and executes the Truera-augmented Sagemaker pipeline
- preprocessing.py: python script containing preprocessing step code
- evaluation.py: python script containing preprocessing step code

## Required inputs:
- data for model training (see abalone sample in /data/input
- connection_string for TruEra deployment configuration
- authorization credentials -- note that basic auth or service account authorization is supported. Token authorization is not supported in current state, as the length of the token exceeds allowable length of Sagemaker pipeline parameters that are passed between pipeline steps. 

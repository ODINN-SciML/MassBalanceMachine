# Model Training with a Custom Loss Function

The following scripts are for the model training stage of the MassBalanceMachine. An XGBoost model with a custom loss function is trained on the stake data.  The following scripts are used in the notebook ```notebooks/model_training_region.ipynb```:

1. ```custom_xgboost_regressor.py``` contains a class extension of the XGBRegressor that takes as an objective loss function a custom loss function specified in ```model_methods.py```.
2. ```model_methods.py``` contains the custom loss function that calculates the MSE for evaluating monthly predictions concerning seasonally or annually aggregated observations. 
3. ```plotting_methods.py``` contains methods to plot cross-validation predictions per fold and aggregated training and testing data for annual and seasonal periods.
4. ```preparation_data.py``` performs different kinds of operations on the data to prepare the data for training, cross-validation, and testing. Each individual data record in the original dataset is converted to multiple data records, depending on the season, such that each record represents a month in the respective season.
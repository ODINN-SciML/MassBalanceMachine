import sys, os

mbm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(mbm_path)  # Add root of repo to import MBM

import warnings
import matplotlib.pyplot as plt
from cmcrameri import cm
import massbalancemachine as mbm
import logging
import torch
import torch.nn as nn
import yaml
import json
import argparse
from skorch.helper import SliceDataset
from datetime import datetime
from skorch.callbacks import EarlyStopping, LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from sklearn.metrics import root_mean_squared_error

from scripts.common import (
    getTrainTestSets,
    _default_test_glaciers,
    _default_train_glaciers,
    _default_input,
    seed_all,
)

from regions.Switzerland.scripts.helpers import get_cmap_hex
from regions.Switzerland.scripts.plots import compute_seasonal_scores, plot_predictions_summary, predVSTruth, plotMeanPred, PlotIndividualGlacierPredVsTruth
from regions.Switzerland.scripts.nn_helpers import plot_training_history, evaluate_model_and_group_predictions

import pdb

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("modelType", type=str, help="Type of model to train")
args = parser.parse_args()
modelType = args.modelType

with open("scripts/netcfg/" + modelType + ".yml") as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

featuresInpModel = params["model"].get("inputs") or _default_input

featuresToRemove = list(set(_default_input) - set(featuresInpModel))
metaData = list(
    set(
        [
            "RGIId",
            "POINT_ID",
            "ID",
            "GLWD_ID",
            "N_MONTHS",
            "MONTHS",
            "PERIOD",
            "GLACIER",
            "YEAR",
            "POINT_LAT",
            "POINT_LON",
        ]
    ).union(set(featuresToRemove))
)

cfg = mbm.SwitzerlandConfig(
    metaData=metaData,
    notMetaDataNotFeatures=["POINT_BALANCE"],
)
seed_all(cfg.seed)




# Plot styles:
path_style_sheet = 'regions/Switzerland/scripts/example.mplstyle'
plt.style.use(path_style_sheet)
colors = get_cmap_hex(cm.batlow, 10)
color_dark_blue = colors[0]
color_pink = '#c51b7d'



if torch.cuda.is_available():
    print("CUDA is available")
    # free_up_cuda()
else:
    print("CUDA is NOT available")


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

train_glaciers = params["training"].get("train_glaciers") or _default_train_glaciers
test_glaciers = params["training"].get("test_glaciers") or _default_test_glaciers

train_set, test_set, data_glamos = getTrainTestSets(
    train_glaciers, test_glaciers, params, cfg, 'CH_wgms_dataset_monthly_NN_nongeo.csv', process=False
)











# Validation and train split:
data_train = train_set['df_X']
data_train['y'] = train_set['y']
dataloader = mbm.dataloader.DataLoader(cfg, data=data_train)

feature_columns = list(
    data_train.columns.difference(cfg.metaData)
    .drop(cfg.notMetaDataNotFeatures)
    .drop("y")
)
assert set(feature_columns) == set(
    featuresInpModel
), f"Asked features are {featuresInpModel} but the one obtained from the dataframe are {feature_columns}"
cfg.setFeatures(feature_columns)


train_itr, val_itr = dataloader.set_train_test_split(test_size=0.2)

# Get all indices of the training and valing dataset at once from the iterators. Once called, the iterators are empty.
train_indices, val_indices = list(train_itr), list(val_itr)

df_X_train = data_train.iloc[train_indices]
y_train = df_X_train['POINT_BALANCE'].values

# Get val set
df_X_val = data_train.iloc[val_indices]
y_val = df_X_val['POINT_BALANCE'].values

assert all(data_train.POINT_BALANCE == train_set["y"])


all_columns = feature_columns + cfg.fieldsNotFeatures
print('Shape of training dataset:', df_X_train[all_columns].shape)
print('Shape of validation dataset:', df_X_val[all_columns].shape)
print('Shape of testing dataset:', test_set["df_X"][all_columns].shape)
print('Running with features:', feature_columns)






early_stop = EarlyStopping(
    monitor='valid_loss',
    patience=20,
    threshold=1e-4,  # Optional: stop only when improvement is very small
)

lr_scheduler_cb = LRScheduler(policy=ReduceLROnPlateau,
                              monitor='valid_loss',
                              mode='min',
                              factor=0.5,
                              patience=5,
                              threshold=0.01,
                              threshold_mode='rel',
                              verbose=True)

dataset = dataset_val = None  # Initialized hereafter


def my_train_split(ds, y=None, **fit_params):
    return dataset, dataset_val


# param_init = {'device': 'cuda:0'}
param_init = {'device': 'cpu'}  # Use CPU for training
nInp = len(feature_columns)






optimType = params["training"].get("optim", "ADAM")
lr = float(params["training"].get("lr", 1e-3))
Nepochs = int(params["training"].get("Nepochs", 1000))
if optimType == "ADAM":
    optim = torch.optim.Adam
elif optimType == "SGD":
    optim = torch.optim.SGD
else:
    raise ValueError(f"Optimizer {optimType} is not supported.")




paramsTOREMOVE = {
    'lr': lr,
    'batch_size': 128,
    'optimizer': optim,
    'optimizer__weight_decay': 1e-05,
}

class NetworkBinding(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

model = mbm.models.buildModel(cfg, params=params)

args = {
    'module': NetworkBinding,
    'nbFeatures': nInp,
    'module__model': model,
    'train_split': my_train_split,
    'batch_size': paramsTOREMOVE['batch_size'],
    'verbose': 1,
    'iterator_train__shuffle': True,
    'lr': paramsTOREMOVE['lr'],
    'max_epochs': Nepochs,
    'optimizer': paramsTOREMOVE['optimizer'],
    'optimizer__weight_decay': paramsTOREMOVE['optimizer__weight_decay'],
    'callbacks': [
        ('early_stop', early_stop),
        ('lr_scheduler', lr_scheduler_cb),
    ]
}


custom_nn = mbm.models.CustomNeuralNetRegressor(cfg, **args, **param_init)








df_X_train_subset = df_X_train
df_X_val_subset = df_X_val
df_X_test_subset = test_set['df_X']



features, metadata = custom_nn._create_features_metadata(df_X_train_subset)

features_val, metadata_val = custom_nn._create_features_metadata(
    df_X_val_subset)

# Define the dataset for the NN
dataset = mbm.data_processing.AggregatedDataset(cfg,
                                                features=features,
                                                metadata=metadata,
                                                targets=y_train)
dataset = mbm.data_processing.SliceDatasetBinding(SliceDataset(dataset, idx=0),
                                                  SliceDataset(dataset, idx=1))
print("train:", dataset.X.shape, dataset.y.shape)

dataset_val = mbm.data_processing.AggregatedDataset(cfg,
                                                    features=features_val,
                                                    metadata=metadata_val,
                                                    targets=y_val)
dataset_val = mbm.data_processing.SliceDatasetBinding(
    SliceDataset(dataset_val, idx=0), SliceDataset(dataset_val, idx=1))
print("validation:", dataset_val.X.shape, dataset_val.y.shape)









TRAIN = True
if TRAIN:
    custom_nn.seed_all()

    print("Training the model...")
    print('Model parameters:')
    for key, value in args.items():
        print(f"{key}: {value}")
    custom_nn.fit(dataset.X, dataset.y)
    # The dataset provided in fit is not used as the datasets are overwritten in the provided train_split function

    # Generate filename with current date
    current_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_filename = f"nn_model_{current_date}.pt"

    plot_training_history(custom_nn, skip_first_n=5)

    # After Training: Best weights are already loaded
    # Save the model
    custom_nn.save_model(model_filename)
    # torch.save(custom_nn.module.state_dict(), model_filename)

    # save params dic
    params_filename = f"nn_params_{current_date}.pkl"

    with open(f"models/{params_filename}", "wb") as f:
        pickle.dump(args, f)
sys.exit(0)





# Load model and set to CPU
model_filename = "nn_model_2025-08-13_16:22:18.pt"  # Replace with actual date if needed
# model_filename = "nn_model_finetuned_winter_2025-07-08.pt"
# read pickle with params
params_filename = "nn_params_2025-08-13_16:22:18.pkl"  # Replace with actual date if needed
with open(f"models/{params_filename}", "rb") as f:
    custom_params = pickle.load(f)

paramsTOREMOVE = custom_params

args = {
    'module': NetworkBinding,
    'nbFeatures': nInp,
    'module__model': model,
    'train_split': my_train_split,
    'batch_size': paramsTOREMOVE['batch_size'],
    'verbose': 1,
    'iterator_train__shuffle': True,
    'lr': paramsTOREMOVE['lr'],
    'max_epochs': 300,
    'optimizer': paramsTOREMOVE['optimizer'],
    'optimizer__weight_decay': paramsTOREMOVE['optimizer__weight_decay'],
    'callbacks': [
        ('early_stop', early_stop),
        ('lr_scheduler', lr_scheduler_cb),
    ]
}

loaded_model = mbm.models.CustomNeuralNetRegressor.load_model(
    cfg,
    model_filename,
    **{
        **args,
        **param_init
    },
)
loaded_model = loaded_model.set_params(device='cpu')
loaded_model = loaded_model.to('cpu')






grouped_ids, scores_NN, ids_NN, y_pred_NN = evaluate_model_and_group_predictions(
    loaded_model, df_X_test_subset, test_set['y'], cfg, mbm)
scores_annual, scores_winter = compute_seasonal_scores(grouped_ids,
                                                       target_col='target',
                                                       pred_col='pred')
fig = plot_predictions_summary(grouped_ids=grouped_ids,
                               scores_annual=scores_annual,
                               scores_winter=scores_winter,
                               predVSTruth=predVSTruth,
                               plotMeanPred=plotMeanPred,
                               color_annual=color_dark_blue,
                               color_winter=color_pink,
                               ax_xlim=(-8, 6),
                               ax_ylim=(-8, 6))





submission_df = grouped_ids[['ID', 'pred']].sort_values(by='ID')
submission_df.rename(columns={'pred': 'POINT_BALANCE'}, inplace=True)
# change 'ID' to string
submission_df['ID'] = submission_df['ID'].astype(str)
# save solution
submission_df.to_csv('submission.csv', index=False)

solution_df = grouped_ids[['ID', 'target']].sort_values(by='ID')
solution_df.rename(columns={'target': 'POINT_BALANCE'}, inplace=True)
# change 'ID' to string
solution_df['ID'] = solution_df['ID'].astype(str)

# save solution
solution_df.to_csv('solution.csv', index=False)

# calculate RMSE
root_mean_squared_error(grouped_ids['target'], grouped_ids['pred'])







gl_per_el = data_glamos[data_glamos.PERIOD == 'annual'].groupby(
    ['GLACIER'])['POINT_ELEVATION'].mean()
gl_per_el = gl_per_el.sort_values(ascending=False)

test_gl_per_el = gl_per_el[test_glaciers].sort_values().index

fig, axs = plt.subplots(3, 3, figsize=(20, 15), sharex=True)

PlotIndividualGlacierPredVsTruth(grouped_ids,
                                 axs=axs,
                                 color_annual=color_dark_blue,
                                 color_winter=color_pink,
                                 custom_order=test_gl_per_el)






grouped_ids_NN_train, scores_NN_train, ids_train, y_pred_train = evaluate_model_and_group_predictions(
    loaded_model, data_train[all_columns], data_train['POINT_BALANCE'].values,
    cfg, mbm)
scores_annual_NN, scores_winter_NN = compute_seasonal_scores(
    grouped_ids_NN_train, target_col='target', pred_col='pred')
fig = plot_predictions_summary(grouped_ids=grouped_ids_NN_train,
                               scores_annual=scores_annual_NN,
                               scores_winter=scores_winter_NN,
                               predVSTruth=predVSTruth,
                               plotMeanPred=plotMeanPred,
                               color_annual=color_dark_blue,
                               color_winter=color_pink,
                               ax_xlim=(-14, 8),
                               ax_ylim=(-14, 8))






train_gl_per_el = gl_per_el[train_glaciers].sort_values().index

fig, axs = plt.subplots(8, 3, figsize=(20, 30), sharex=False)

PlotIndividualGlacierPredVsTruth(grouped_ids_NN_train,
                                 axs=axs,
                                 color_annual=color_dark_blue,
                                 color_winter=color_pink,
                                 custom_order=train_gl_per_el,
                                 add_text=True,
                                 ax_xlim=None)



# Copied up to the "Extrapolate in space" section

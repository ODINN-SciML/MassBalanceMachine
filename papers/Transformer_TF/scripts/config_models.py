PARAMS_LSTM = {
    "Fm": 8,
    "Fs": 3,
    "hidden_size": 96,
    "num_layers": 2,
    "bidirectional": False,
    "dropout": 0.2,
    "static_layers": 1,
    "static_hidden": 128,
    "static_dropout": 0.3,
    "lr": 0.0005,
    "weight_decay": 1e-05,
    "loss_name": "neutral",
    "two_heads": False,
    "head_dropout": 0.0,
    "loss_spec": None,
}

# Transformer used in all experiments of the paper: best configuration of the
# multi-region grid search in `3.1 Pooled_model_grid_search` (see GS_* below)
PARAMS_TRANSFORMER = {
    "Fm": 8,
    "Fs": 3,
    "d_model": 64,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.0,
    "static_layers": 2,
    "static_hidden": 128,
    "static_dropout": 0.1,
    "head_dropout": 0.0,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "two_heads": False,
    "loss_spec": None,
    "T_max": 32,
}

MODEL_DATE = "2026-06-10"  # date tag in the file names of the trained models
N_EPOCHS = 150  # maximum training epochs (early stopping on the validation loss)

# --- Hyperparameter search (3.1 Pooled_model_grid_search) ---
# Random search: GS_N_TRIALS configurations sampled (seed cfg.seed) from the grid
# in 3.1, skipping combinations where nhead does not divide d_model or
# dim_feedforward < d_model. Each is trained on the source regions (CH, NOR, ISL,
# FR) for up to GS_MAX_EPOCHS epochs. Validation data: the test glaciers of the
# within-region splits of 2.3 (about 16-24% of each region's measurements).
# Selection rule: the configuration with the lowest pooled validation loss.
GS_N_TRIALS = 150
GS_MAX_EPOCHS = 150
GS_SELECTION_METRIC = "valid_loss"  # lowest value wins
GS_LOG = "FD_MODEL/GS_results/pooled_tf_gs_2026-06-12.csv"  # under cfg.dataPath/cache/TF_Europe

# --- Fine-tuning of the multi-region model (3.2-3.4) ---
# FT_FRAC of the measurement IDs of each target region (winter and annual drawn
# separately) are used for fine-tuning, the rest for testing. Repeated
# FT_N_REPEATS times with seeds cfg.seed + rep (rep = 0..FT_N_REPEATS-1; with
# cfg.seed = 20 these are 20-24). Within the FT data, an 80/20 train/validation
# split is used, with early stopping (patience 10) and the best-validation
# checkpoint restored.
FT_FRAC = 0.10
FT_N_REPEATS = 5

PARAMS_FT = {
    "epochs": 50,
    "lr_factor": 0.1,
}  # max epochs; lr = PARAMS_TRANSFORMER lr x lr_factor

PARAMS_DAN = {
    "grl_lambda": 1.0,
    "lr_backbone": 5e-05,
    "lr_domain": 0.0001,
    "epochs": 60,
    "dan_alpha": 0.05,
    "mix_ratio_ft": 0.5,
}

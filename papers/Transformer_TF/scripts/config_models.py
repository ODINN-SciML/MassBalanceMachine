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

MODEL_DATE = "2026-06-10"
N_EPOCHS = 150

PARAMS_FT = {"epochs": 50, "lr_factor": 0.1}

PARAMS_DAN = {
    "grl_lambda": 1.0,
    "lr_backbone": 5e-05,
    "lr_domain": 0.0001,
    "epochs": 60,
    "dan_alpha": 0.05,
    "mix_ratio_ft": 0.5,
}

# Get the best Transformer parameters:
# gs_logs_dir = Path(cfg.dataPath) / path_cache / "TF_REGION/logs/Transformer_GS"
# logs_gs_transformer = pd.read_csv(
#     os.path.join(gs_logs_dir, "transformer_gs_pooled_2026-05-20.csv"))
# logs_gs_transformer.sort_values("valid_loss")

# # best by validation loss
# best_row = logs_gs_transformer.sort_values("valid_loss").iloc[0]
# print("Best config:")
# print(best_row[[
#     "d_model", "nhead", "num_layers", "dim_feedforward", "dropout",
#     "static_layers", "static_hidden", "static_dropout", "head_dropout", "lr",
#     "weight_decay", "valid_loss", "val_rmse_a", "val_rmse_w"
# ]])

# best_params_gs = {
#     "Fm":
#     int(best_row["Fm"]),
#     "Fs":
#     int(best_row["Fs"]),
#     "d_model":
#     int(best_row["d_model"]),
#     "nhead":
#     int(best_row["nhead"]),
#     "num_layers":
#     int(best_row["num_layers"]),
#     "dim_feedforward":
#     int(best_row["dim_feedforward"]),
#     "dropout":
#     float(best_row["dropout"]),
#     "static_layers":
#     int(best_row["static_layers"]),
#     "static_hidden": (None if pd.isna(best_row["static_hidden"]) else int(
#         best_row["static_hidden"])),
#     "static_dropout": (None if pd.isna(best_row["static_dropout"]) else float(
#         best_row["static_dropout"])),
#     "head_dropout":
#     float(best_row["head_dropout"]),
#     "lr":
#     float(best_row["lr"]),
#     "weight_decay":
#     float(best_row["weight_decay"]),
#     "two_heads":
#     False,
#     "loss_spec":
#     None,
#     "T_max":
#     32,
# }

# print("\nbest_params_gs:")
# for k, v in best_params_gs.items():
#     print(f"  {k}: {v}")

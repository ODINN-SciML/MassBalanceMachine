import os
import xarray as xr


# Used by papers/Transformer_TF/scripts/plotting/pooled_pred.py and 3.3 Pooled_model_pred
def _load_lstm_ds(glacier, year, path_pred_lstm, period, apply_smoothing_fn=None):
    zpath = os.path.join(path_pred_lstm, glacier, f"{glacier}_{year}_{period}.zarr")
    if not os.path.exists(zpath):
        return None
    ds = xr.open_zarr(zpath)
    if apply_smoothing_fn is not None:
        ds = apply_smoothing_fn(ds)
    return ds

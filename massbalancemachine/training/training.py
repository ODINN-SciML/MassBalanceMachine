import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import r2_score
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import numpy as np
import datetime
import heapq
import matplotlib.pyplot as plt
import pandas as pd
import glob
import warnings
import json
import git
import time

from plots import predVSTruth, predVSTruthGlacierWide
from models.TorchNeuralNetworkRegressor import (
    aggrPredict,
    aggrMetadataId,
    aggrPredictGlwd,
    aggrMetadataGlwdId,
)


def check_async_transfer_compatibility(geodataloader):
    if geodataloader.device.type == "cuda":
        try:
            tmp = torch.zeros(3).pin_memory()
            tmp = tmp.to(geodataloader.device, non_blocking=True)
            async_transfer = True
            del tmp
        except Exception as e:
            print(e)
            warnings.warn("Error while trying to setup async data transfer on GPU.")
            async_transfer = False
    else:
        async_transfer = False
    return async_transfer


def in_jupyter_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        pass
    return False


_inJupyterNotebook = in_jupyter_notebook()
if _inJupyterNotebook:
    import tqdm.notebook as tqdm
else:
    import tqdm

_default_log_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs/")
)
_criterionVal = ["lossVal", "lossStake", "lossValGeo", "mse", "rmse", "mae", "pearson"]
_maxCriterion = ["pearson"]  # Criterion for which higher is better


def compute_stake_loss(model, stakes, metadata, point_balance, returnPred=False):
    """
    Computes the stake loss term.

    Args:
        model (CustomTorchNeuralNetRegressor): Model to evaluate.
        stakes (torch.Tensor): Stake measurements features returned by the geodetic
            dataloader.
        metadata (pd.DataFrame): Metadata returned by the geodetic dataloader.
        point_balance (torch.Tensor): Mass balance value of each stake measurement.
            It is returned by the geodetic dataloader.
        returnPred (bool): Whether to return the prediction and the target in a
            dictionary. Default is False.

    Returns a scalar torch value that corresponds to the stake loss term and
    optionally statistics in a dictionary.
    """
    idAggr = metadata["ID"].values
    int_id, unique_id = pd.factorize(idAggr)

    # Make prediction
    pred = model.forward(stakes)[:, 0]

    trueMean = torch.zeros(
        (len(np.unique(idAggr)),), device=pred.device, dtype=pred.dtype
    )

    # Aggregate per stake and periods
    aggrPredict(point_balance, int_id, reduce="mean", out=trueMean)
    predSum = aggrPredict(pred, int_id)

    mse = nn.functional.mse_loss(predSum, trueMean, reduction="mean")
    ret = {}
    if returnPred:
        ret["target"] = trueMean.detach().cpu()
        ret["pred"] = predSum.detach().cpu()
    return mse, ret, int_id


# TODO: time aggregation!
def timeWindowGeodeticLoss(
    predSumAnnualGlwd, geoTarget, errGeoTarget, metadataAggrYear, geod_periods
):
    """
    Given glacier wide mass balance values for different years, this function
    computes the predicted geodetic MB values over different time windows and then
    computes the loss term for each of these windows.

    Args:
        predSumAnnualGlwd (torch.Tensor): Predicted MB values for different years.
        geoTarget (torch.Tensor): Ground truth MB values for the different time windows.
        errGeoTarget (torch.Tensor): 1 sigma error of the ground truth MB values for the
            different time windows.
        metadataAggrYear (pd.DataFrame): Year associated to each prediction. Must be
            of the same length as `predSumAnnualGlwd`.
        geod_periods (dict of tuple of ints): Dictionary containing the time windows
            as tuples with 2 integer values which are the start and end years. Must
            be of the same size as `geoTarget`.

    Returns a torch.Tensor that contains the geodetic loss terms for each of the
    time windows.
    """
    assert len(geoTarget) == len(
        geod_periods
    ), f"Size of the ground truth is {geoTarget.shape} but doesn't match with the number of geodetic periods which is {len(geod_periods)}"
    yearsPred = metadataAggrYear.YEAR.values

    geodetic_MB_err = torch.zeros(len(geod_periods))
    for e, (start_year, end_year) in enumerate(geod_periods):
        geodetic_range = range(
            start_year, end_year
        )  # end_year is 2021 when the end date is 2021-01-01

        # Ensure years exist in index before selection
        valid_years = [yr for yr in geodetic_range if yr in yearsPred]
        valid_years = []
        indSlice = []
        for yr in geodetic_range:
            if yr in yearsPred:
                valid_years.append(yr)
                indSlice.append(np.argwhere(yearsPred == yr)[0, 0])
        if valid_years:
            # geodetic_MB_err[e] = (
            #     torch.mean(predSumAnnualGlwd[indSlice]) - geoTarget[e]
            # ) ** 2
            geodetic_MB_err[e] = (
                torch.clamp(
                    torch.abs(torch.mean(predSumAnnualGlwd[indSlice]) - geoTarget[e])
                    - errGeoTarget[e],
                    min=0,
                )
                ** 2
            )
        else:
            geodetic_MB_err[e] = np.nan  # Handle missing years
    return geodetic_MB_err


def predict_monthly_gridded(model, geoGrid, metadata):
    # Make prediction
    pred = model.forward(geoGrid)[:, 0]
    return pred


def predict_annual_gridded(model, geoGrid, metadata):
    pred = predict_monthly_gridded(model, geoGrid, metadata)

    idAggr = metadata["ID"].values
    int_id, unique_id = pd.factorize(idAggr)
    # TODO: update here and everywhere else needed
    metadata = metadata.assign(ID_int=int_id)

    # Aggregate per point on the grid
    grouped_ids = aggrMetadataId(metadata, "ID_int")
    predSumAnnual = aggrPredict(pred, metadata["ID_int"].values)

    return grouped_ids, predSumAnnual


def eval_geodetic(model, geo_dataloader, return_grid_pred=[]):
    geoPred = {}
    geoTarget = {}
    geoErr = {}
    return_monthly = "monthly" in return_grid_pred
    return_annual = "annual" in return_grid_pred
    df_gridded_monthly = pd.DataFrame() if return_monthly else None
    df_gridded_annual = pd.DataFrame() if return_annual else None
    with torch.no_grad():

        async_transfer = check_async_transfer_compatibility(geo_dataloader)

        with tqdm.tqdm(
            geo_dataloader.glaciersGeo(), total=geo_dataloader.lenGeo()
        ) as pbar:

            glacier_iter = iter(geo_dataloader.glaciersGeo())
            try:
                current_g = next(glacier_iter)
            except StopIteration:
                current_g = None

            # geo future loaded at the first iteration
            current_geo_future = None
            if current_g is not None:
                current_geo_future = geo_dataloader.submit_geo(current_g)

            batch_idx = 0
            while current_g is not None:

                # Look ahead and start loading geo for the next iteration
                try:
                    next_g = next(glacier_iter)
                except StopIteration:
                    next_g = None

                next_geo_future = None
                if next_g is not None:
                    next_geo_future = geo_dataloader.submit_geo(next_g)

                # pbar = tqdm.tqdm(geo_dataloader.glaciersGeo(), total=geo_dataloader.lenGeo())
                # for g in pbar:
                pbar.set_description(
                    "Making geodetic pred for %s" % (current_g), refresh=True
                )
                pbar.update(1)
                # consume prefetched current batch
                if current_geo_future is not None:
                    geoGrid, metadata, ygeo, errgeo, precomputed_meta = (
                        current_geo_future.result()
                    )
                else:
                    geoGrid, metadata, ygeo, errgeo, precomputed_meta = (
                        geo_dataloader.geo(current_g)
                    )

                geoGrid = geoGrid.to(geo_dataloader.device, non_blocking=async_transfer)
                geod_periods = geo_dataloader.periods_per_glacier[current_g]
                geoPred[current_g] = (
                    predict_geo(model, geoGrid, metadata, ygeo, geod_periods)
                    .cpu()
                    .item()
                )
                geoTarget[current_g] = ygeo.item()
                geoErr[current_g] = errgeo.item()

                if return_annual:
                    grouped_ids, predSumAnnual = predict_annual_gridded(
                        model, geoGrid, metadata
                    )
                    grouped_ids["pred"] = predSumAnnual.cpu()
                    df_gridded_annual = pd.concat([df_gridded_annual, grouped_ids])
                if return_monthly:
                    predMonthly = predict_monthly_gridded(model, geoGrid, metadata)
                    metadata["pred"] = predMonthly.cpu()
                    df_gridded_monthly = pd.concat([df_gridded_monthly, metadata])

                # Shift pipeline
                current_g = next_g
                current_geo_future = next_geo_future
                batch_idx += 1

    dict_df_gridded = {}
    if return_monthly:
        dict_df_gridded["monthly"] = df_gridded_monthly
    if return_annual:
        dict_df_gridded["annual"] = df_gridded_annual
    return geoPred, geoTarget, geoErr, dict_df_gridded


def predict_geo(model, geoGrid, metadata, ygeo, geod_periods):
    # TODO: optimize this section
    # Make prediction and aggregate per point on the grid
    grouped_ids, predSumAnnual = predict_annual_gridded(model, geoGrid, metadata)

    # Create ID to aggregate glacier wide
    idGlwdAggr = grouped_ids["GLWD_ID"].values
    int_id_glwd, _ = pd.factorize(idGlwdAggr)
    grouped_ids = grouped_ids.assign(GLWD_ID_int=int_id_glwd)

    # Aggregate glacier wide
    metadataAggrYear = aggrMetadataGlwdId(grouped_ids, "GLWD_ID_int")
    predSumAnnualGlwd = aggrPredictGlwd(
        predSumAnnual, grouped_ids["GLWD_ID_int"].values
    )

    # TODO: remove the loop since the grid should already correspond to the geodetic period

    assert len(ygeo) == len(
        geod_periods
    ), f"Size of the ground truth is {ygeo.shape} but doesn't match with the number of geodetic periods which is {len(geod_periods)}"
    yearsPred = metadataAggrYear.YEAR.values

    geodetic_MB_pred = torch.zeros(len(geod_periods), device=predSumAnnualGlwd.device)
    for e, (start_year, end_year) in enumerate(geod_periods):
        geodetic_range = range(
            start_year, end_year
        )  # end_year is 2021 when the end date is 2021-01-01

        # Ensure years exist in index before selection
        valid_years = [yr for yr in geodetic_range if yr in yearsPred]
        valid_years = []
        indSlice = []
        for yr in geodetic_range:
            if yr in yearsPred:
                valid_years.append(yr)
                indSlice.append(np.argwhere(yearsPred == yr)[0, 0])
        if valid_years:
            geodetic_MB_pred[e] = torch.mean(predSumAnnualGlwd[indSlice])
        else:
            geodetic_MB_pred[e] = np.nan  # Handle missing years
    return geodetic_MB_pred


def compute_geo_loss(
    model, geoGrid, metadata, ygeo, errgeo, geod_periods, precomputed_meta
):
    # TODO: update docstring
    """
    Computes the geodetic loss term.

    Args:
        model (CustomTorchNeuralNetRegressor): Model to evaluate.
        geoGrid (torch.Tensor): Geodetic features returned by the geodetic dataloader.
        metadata (pd.DataFrame): Metadata returned by the geodetic dataloader.
        ygeo (torch.Tensor): Geodetic mass balance values. It is returned by the
            geodetic dataloader.
        errgeo (torch.Tensor): 1 sigma error of the geodetic mass balance values.
            It is also returned by the geodetic dataloader.

    Returns a scalar torch value that corresponds to the geodetic loss term.
    """
    # Make prediction
    with record_function("geo_forward"):
        pred = model.forward(geoGrid)[:, 0]

    with record_function("aggregation_ID"):
        # idAggr = metadata["ID"].values
        # int_id, unique_id = pd.factorize(idAggr)
        # metadata = metadata.assign(ID_int=int_id)

        # Aggregate per point on the grid
        # grouped_ids = aggrMetadataId(metadata, "ID_int")
        grouped_ids = precomputed_meta["grouped_ids"]

        idAggr = metadata[
            "ID_int"
        ].values  # TODO: could be transfered to the GPU in advance (async in the dataloader)
        nunique = precomputed_meta["nunique_ids"]
        predSumAnnual = torch.zeros((nunique,), device=pred.device, dtype=pred.dtype)
        aggrPredict(pred, idAggr, out=predSumAnnual)

    with record_function("aggregation_GLWD_ID"):

        # Aggregate glacier wide
        metadataAggrYear = precomputed_meta["grouped_glwd_ids"]
        idAggr = grouped_ids[
            "GLWD_ID_int"
        ].values  # TODO: could be transfered to the GPU in advance (async in the dataloader)
        nunique = precomputed_meta["nunique_glwd_ids"]
        predSumAnnualGlwd = torch.zeros(
            (nunique,), device=pred.device, dtype=pred.dtype
        )
        aggrPredictGlwd(predSumAnnual, idAggr, out=predSumAnnualGlwd)

    # Compute the geodetic MB for the different time windows
    with record_function("timeWindowGeodeticLoss"):
        lossGeo = timeWindowGeodeticLoss(
            predSumAnnualGlwd,
            ygeo,
            errgeo,
            metadataAggrYear,
            geod_periods,
        )

    return lossGeo.mean()  # Compute mean of the different time window scores


def scores(pred: torch.Tensor, target: torch.Tensor):
    mse = nn.functional.mse_loss(pred, target, reduction="mean")
    rmse = mse.sqrt()
    mae = nn.functional.l1_loss(pred, target, reduction="mean")
    pearson_corr = torch.corrcoef(
        torch.cat((pred[:, None], target[:, None]), axis=1).T
    )[0, 1]
    r2 = r2_score(pred, target)
    bias = torch.mean(pred - target)
    return mse, rmse, mae, pearson_corr, r2, bias


def assessOnTest(log_dir, model, geodataloader_test, light=False):
    targetAll = torch.zeros(0)
    predAll = torch.zeros(0)
    periodAll = np.zeros(
        0, dtype=np.array(list(geodataloader_test.periodToInt.keys())).dtype
    )  # Initialize with correct dtype
    for g in geodataloader_test.glaciers():
        stakes, metadata, point_balance = geodataloader_test.stakes(g)
        stakes = torch.tensor(stakes.astype(np.float32)).to(geodataloader_test.device)
        point_balance = torch.tensor(point_balance.astype(np.float32)).to(
            geodataloader_test.device
        )
        l, ret, int_id = compute_stake_loss(
            model, stakes, metadata, point_balance, returnPred=True
        )
        targetAll = torch.concatenate((targetAll, ret["target"]))
        predAll = torch.concatenate((predAll, ret["pred"]))
        metadata = metadata.assign(ID_int=int_id)
        grouped_ids = metadata.groupby("ID_int").agg({"PERIOD": "first"})
        periodAll = np.concatenate((periodAll, np.array(grouped_ids["PERIOD"].values)))
    mse, rmse, mae, pearson_corr, r2, bias = scores(predAll, targetAll)

    indAnnual = np.argwhere(periodAll == "annual")[:, 0]
    indWinter = np.argwhere(periodAll == "winter")[:, 0]
    indSummer = np.argwhere(periodAll == "summer")[:, 0]
    predAnnual = predAll[indAnnual]
    targetAnnual = targetAll[indAnnual]
    predWinter = predAll[indWinter]
    targetWinter = targetAll[indWinter]
    predSummer = predAll[indSummer]
    targetSummer = targetAll[indSummer]
    mse_annual, rmse_annual, mae_annual, pearson_corr_annual, r2_annual, bias_annual = (
        scores(predAnnual, targetAnnual)
    )
    mse_winter, rmse_winter, mae_winter, pearson_corr_winter, r2_winter, bias_winter = (
        scores(predWinter, targetWinter)
    )
    if len(targetSummer) > 0:
        (
            mse_summer,
            rmse_summer,
            mae_summer,
            pearson_corr_summer,
            r2_summer,
            bias_summer,
        ) = scores(predSummer, targetSummer)

    # Make plots
    plot_pred_vs_obs(log_dir, targetAll, predAll, {"rmse": rmse, "mae": mae})

    if not light:
        # Geodetic prediction
        geoPred, geoTarget, geoErr, _ = eval_geodetic(model, geodataloader_test)
        fig = predVSTruthGlacierWide(
            geoTarget, geoPred, geoErr, title="Glacier wide MB on test"
        )
        plt.savefig(os.path.join(log_dir, "geodetic_test.png"))
        plt.close(fig)

    stats = {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "pearson": pearson_corr.item(),
        "r2": r2.item(),
        "bias": bias.item(),
        "mse_annual": mse_annual.item(),
        "rmse_annual": rmse_annual.item(),
        "mae_annual": mae_annual.item(),
        "pearson_annual": pearson_corr_annual.item(),
        "r2_annual": r2_annual.item(),
        "bias_annual": bias_annual.item(),
        "mse_winter": mse_winter.item(),
        "rmse_winter": rmse_winter.item(),
        "mae_winter": mae_winter.item(),
        "pearson_winter": pearson_corr_winter.item(),
        "r2_winter": r2_winter.item(),
        "bias_winter": bias_winter.item(),
    }
    if len(targetSummer) > 0:
        stats["mse_summer"] = mse_summer.item()
        stats["rmse_summer"] = rmse_summer.item()
        stats["mae_summer"] = mae_summer.item()
        stats["pearson_summer"] = pearson_corr_summer.item()
        stats["r2_summer"] = r2_summer.item()
        stats["bias_summer"] = bias_summer.item()
    return stats


def plot_pred_vs_obs(log_dir, target, pred, scores):
    grouped_ids = pd.DataFrame({"target": target.numpy(), "pred": pred.numpy()})
    fig = predVSTruth(
        grouped_ids, scores=scores, marker="o", title="NN on test", alpha=0.5
    )
    plt.savefig(os.path.join(log_dir, "predVsObs.png"))
    plt.close(fig)


def loadBestModel(log_dir, model):
    files = glob.glob(os.path.join(log_dir, "model_epoch*.pt"))
    best = None
    bestVal = None
    bestEpoch = None
    for i, f in enumerate(files):
        s = os.path.basename(f).replace("model_epoch", "").replace(".pt", "")
        epoch = s.split("_")[0]
        val = s.split("_")[1]
        for metric in _criterionVal:
            if metric in val:
                higherBetter = metric in _maxCriterion
                val = val.replace(metric, "")
        if bestVal is None:
            best = i
            bestVal = val
            bestEpoch = epoch
        elif val == bestVal and (epoch > bestEpoch):
            warnings.warn(
                "Two models have exactly the same validation score. Taking the one with the largest epoch number."
            )
            best = i
            bestVal = val
            bestEpoch = epoch
        elif higherBetter and val > bestVal:
            best = i
            bestVal = val
            bestEpoch = epoch
        elif not higherBetter and val < bestVal:
            best = i
            bestVal = val
            bestEpoch = epoch
        val = float(val)
    if best is None:
        raise Exception("No model found.")
    model.load_state_dict(torch.load(files[best], weights_only=True))
    return files[best]


def train_geo(
    model,
    geodataloader,
    optim,
    params,
    scheduler=None,
    geodataloader_test=None,
    timeExec=False,
    useProfiler=False,
):
    """
    Train a model with both stake measurements and geodetic data.

    Args:
        model (CustomTorchNeuralNetRegressor): Model to train.
        geodataloader (GeoDataLoader): Dataloader that provides both stake
            measurements and geodetic data.
        optim (PyTorch optimizer): Optimizer instance to use.
        params (dict): Model and training hyper-parameters.
        scheduler (PyTorch LR scheduler): The learning rate scheduler (optional).
        geodataloader_test (GeoDataLoader): Optional dataloader that provides both
            stake measurements and geodetic data on the test set.
        timeExec (bool): Whether to evaluate loading and inference time.
        useProfiler (bool): Whether to profile the code.
    """
    Nepochs = params["training"]["Nepochs"]
    wGeo = params["training"]["wGeo"]
    freqVal = params["training"]["freqVal"]
    bestModelCriterion = params["training"]["bestModelCriterion"]
    assert bestModelCriterion in _criterionVal
    scalingStakes = params["training"]["scalingStakes"]
    assert scalingStakes in ["meas", "glacier"]
    iterPerEpoch = len(geodataloader)
    nColsProgressBar = 500 if _inJupyterNotebook else 100

    # Setup logging
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_suffix = params["training"]["log_suffix"]
    if log_suffix != "":
        log_suffix = "_" + log_suffix
    if params["training"]["log_dir"] is None:
        log_dir = os.path.join(
            _default_log_dir,
            f"geo_{run_name}{log_suffix}",
        )
    else:
        log_dir = params["training"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # Save params
    repo = git.Repo(search_parent_directories=True)
    params["commit_hash"] = repo.head.object.hexsha
    with open(os.path.join(log_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4, sort_keys=True)

    print(f"Training over {Nepochs} epochs and logging in {log_dir}")

    writer = SummaryWriter(log_dir=log_dir)

    statsTraining = {
        "loss": [],
        "lossStake": [],
        "lossGeo": [],
        "lr": [],
    }
    valMetrics = ["mse", "rmse", "mae", "pearson", "r2", "bias"]
    statsVal = {
        "lossVal": [],
        "lossValStake": [],
        "lossValGeo": [],
    }
    for suffix in ["", "_annual", "_winter", "_summer"]:
        for metric in valMetrics:
            statsVal[metric + suffix] = []
    top_models = []  # Heap of (val_loss, filepath) to store top 5 models

    async_transfer = check_async_transfer_compatibility(geodataloader)

    try:

        for epoch in tqdm.tqdm(
            range(Nepochs),
            desc="Epochs",
            position=0,
            leave=True,
            ncols=nColsProgressBar,
        ):

            avg_training_loss = 0.0
            model.train()
            with tqdm.tqdm(
                total=iterPerEpoch,
                desc=f"Batch",
                position=1,
                leave=False,
                ncols=nColsProgressBar,
            ) as batch_bar:

                glacier_iter = iter(geodataloader.glaciers())
                try:
                    current_g = next(glacier_iter)
                except StopIteration:
                    current_g = None

                # geo future loaded at the first iteration
                current_geo_future = None
                if (
                    current_g is not None
                    and wGeo > 0
                    and geodataloader.hasGeo(current_g)
                ):
                    current_geo_future = geodataloader.submit_geo(current_g)

                batch_idx = 0
                while current_g is not None:
                    # for batch_idx, g in enumerate(geodataloader.glaciers()):

                    prof_ctx = (
                        profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True,
                            profile_memory=True,
                        )
                        if useProfiler
                        else nullcontext()
                    )
                    with prof_ctx as prof:

                        optim.zero_grad()

                        # Look ahead and start loading geo for the next iteration
                        try:
                            next_g = next(glacier_iter)
                        except StopIteration:
                            next_g = None

                        next_geo_future = None
                        if (
                            next_g is not None
                            and wGeo > 0
                            and geodataloader.hasGeo(next_g)
                        ):
                            next_geo_future = geodataloader.submit_geo(next_g)

                        if timeExec:
                            torch.cuda.synchronize()
                            st = time.time()
                        stakes, metadata, point_balance = geodataloader.stakes(
                            current_g
                        )
                        stakes = torch.tensor(stakes.astype(np.float32)).to(
                            geodataloader.device
                        )
                        point_balance = torch.tensor(
                            point_balance.astype(np.float32)
                        ).to(geodataloader.device)
                        if timeExec:
                            torch.cuda.synchronize()
                            stakesDataloaderTime = time.time() - st
                            st = time.time()
                        with record_function("stake_forward"):
                            lossStake, _, _ = compute_stake_loss(
                                model, stakes, metadata, point_balance
                            )
                        if timeExec:
                            torch.cuda.synchronize()
                            stakesInferenceTime = time.time() - st

                        valScalingStakes = (
                            stakes.shape[0] if scalingStakes == "meas" else 1.0
                        )
                        lossStake = lossStake * valScalingStakes
                        if wGeo > 0 and geodataloader.hasGeo(current_g):
                            if timeExec:
                                torch.cuda.synchronize()
                                st = time.time()

                            # consume prefetched current batch
                            if current_geo_future is not None:
                                geoGrid, metadata, ygeo, errgeo, precomputed_meta = (
                                    current_geo_future.result()
                                )
                            else:
                                geoGrid, metadata, ygeo, errgeo, precomputed_meta = (
                                    geodataloader.geo(current_g)
                                )

                            # geoGrid, metadata, ygeo, errgeo = geodataloader.geo(g)
                            # Transfer below takes 20 to 40ms
                            # geoGrid = torch.tensor(geoGrid.astype(np.float32)).to(
                            #     geodataloader.device
                            # )
                            # ygeo = torch.tensor(ygeo.astype(np.float32)).to(
                            #     geodataloader.device
                            # )
                            # errgeo = torch.tensor(errgeo.astype(np.float32)).to(
                            #     geodataloader.device
                            # )
                            geoGrid = geoGrid.to(
                                geodataloader.device, non_blocking=async_transfer
                            )
                            ygeo = ygeo.to(
                                geodataloader.device, non_blocking=async_transfer
                            )
                            errgeo = errgeo.to(
                                geodataloader.device, non_blocking=async_transfer
                            )
                            if timeExec:
                                torch.cuda.synchronize()
                                geoDataloaderTime = time.time() - st
                            geod_periods = geodataloader.periods_per_glacier[current_g]
                            if timeExec:
                                torch.cuda.synchronize()
                                st = time.time()
                            lossGeo = compute_geo_loss(
                                model,
                                geoGrid,
                                metadata,
                                ygeo,
                                errgeo,
                                geod_periods,
                                precomputed_meta,
                            )
                            if timeExec:
                                torch.cuda.synchronize()
                                geoInferenceTime = time.time() - st

                            loss = lossStake + wGeo * lossGeo
                        else:
                            lossGeo = torch.tensor(torch.nan)
                            loss = lossStake
                            geoDataloaderTime = 0.0
                            geoInferenceTime = 0.0

                        if timeExec:
                            torch.cuda.synchronize()
                            st = time.time()
                        with record_function("backward"):
                            loss.backward()
                            optim.step()
                        if timeExec:
                            torch.cuda.synchronize()
                            backwardOptimTime = time.time() - st

                    if useProfiler:
                        prof.export_chrome_trace(f"trace_{batch_idx}.json")
                        if batch_idx == 3:
                            assert False

                    # Statistics
                    lr = optim.param_groups[0]["lr"]
                    statsTraining["lossStake"].append(lossStake.item())
                    statsTraining["lossGeo"].append(lossGeo.item())
                    statsTraining["loss"].append(loss.item())
                    statsTraining["lr"].append(lr)
                    avg_training_loss += statsTraining["loss"][-1]

                    # Log to TensorBoard
                    globalStep = iterPerEpoch * epoch + batch_idx
                    # Metrics
                    writer.add_scalar("Loss/train", loss.item(), globalStep)
                    writer.add_scalar("LossStake/train", lossStake.item(), globalStep)
                    writer.add_scalar("LossGeo/train", lossGeo.item(), globalStep)
                    writer.add_scalar("Step", optim.param_groups[0]["lr"], globalStep)

                    # Timing
                    if timeExec:
                        writer.add_scalar(
                            "TimeLoading/stakes", stakesDataloaderTime, globalStep
                        )
                        writer.add_scalar(
                            "TimeLoading/geo", geoDataloaderTime, globalStep
                        )
                        writer.add_scalar(
                            "TimeInference/stakes", stakesInferenceTime, globalStep
                        )
                        writer.add_scalar(
                            "TimeInference/geo", geoInferenceTime, globalStep
                        )
                        writer.add_scalar(
                            "TimeBackwardOptim", backwardOptimTime, globalStep
                        )

                    # Shift pipeline
                    current_g = next_g
                    current_geo_future = next_geo_future
                    batch_idx += 1

                    batch_bar.set_postfix(
                        loss=f"{statsTraining['loss'][-1]:.4f}",
                        lossStake=f"{statsTraining['lossStake'][-1]:.4f}",
                        lossGeo=f"{statsTraining['lossGeo'][-1]:.4f}",
                    )
                    batch_bar.update(1)

            avg_training_loss /= iterPerEpoch
            if scheduler is not None:
                scheduler.step()

            if freqVal and geodataloader.lenVal() > 0:
                model.eval()
                with torch.no_grad():
                    cntStake = 0
                    cntGeo = 0
                    lossStake = 0.0
                    lossGeo = 0.0
                    targetAll = torch.zeros(0)
                    predAll = torch.zeros(0)
                    periodAll = np.zeros(
                        0, dtype=np.array(list(geodataloader.periodToInt.keys())).dtype
                    )  # Initialize with correct dtype

                    glacier_iter = iter(geodataloader.glaciersVal())
                    try:
                        current_g = next(glacier_iter)
                    except StopIteration:
                        current_g = None

                    # geo future loaded at the first iteration
                    current_geo_future = None
                    if (
                        current_g is not None
                        and wGeo > 0
                        and geodataloader.hasGeo(current_g)
                    ):
                        current_geo_future = geodataloader.submit_geo(current_g)

                    batch_idx = 0
                    while current_g is not None:
                        # for g in geodataloader.glaciersVal():

                        # Look ahead and start loading geo for the next iteration
                        try:
                            next_g = next(glacier_iter)
                        except StopIteration:
                            next_g = None

                        next_geo_future = None
                        if (
                            next_g is not None
                            and wGeo > 0
                            and geodataloader.hasGeo(next_g)
                        ):
                            next_geo_future = geodataloader.submit_geo(next_g)

                        stakes, metadata, point_balance = geodataloader.stakesVal(
                            current_g
                        )
                        stakes = torch.tensor(stakes.astype(np.float32)).to(
                            geodataloader.device
                        )
                        point_balance = torch.tensor(
                            point_balance.astype(np.float32)
                        ).to(geodataloader.device)
                        l, ret, int_id = compute_stake_loss(
                            model, stakes, metadata, point_balance, returnPred=True
                        )
                        target = ret["target"]
                        pred = ret["pred"]
                        targetAll = torch.concatenate((targetAll, target))
                        predAll = torch.concatenate((predAll, pred))
                        metadata = metadata.assign(ID_int=int_id)
                        grouped_ids = metadata.groupby("ID_int").agg(
                            {"PERIOD": "first"}
                        )
                        periodAll = np.concatenate(
                            (periodAll, np.array(grouped_ids["PERIOD"].values))
                        )

                        valScalingStakes = (
                            stakes.shape[0] if scalingStakes == "meas" else 1.0
                        )
                        l = l * valScalingStakes

                        lossStake += l

                        if wGeo > 0 and geodataloader.hasGeo(current_g):

                            # consume prefetched current batch
                            if current_geo_future is not None:
                                geoGrid, metadata, ygeo, errgeo, precomputed_meta = (
                                    current_geo_future.result()
                                )
                            else:
                                geoGrid, metadata, ygeo, errgeo, precomputed_meta = (
                                    geodataloader.geo(current_g)
                                )

                            # geoGrid, metadata, ygeo, errgeo = geodataloader.geo(g)
                            # geoGrid = torch.tensor(geoGrid.astype(np.float32)).to(
                            #     geodataloader.device
                            # )
                            # ygeo = torch.tensor(ygeo.astype(np.float32)).to(
                            #     geodataloader.device
                            # )
                            # errgeo = torch.tensor(errgeo.astype(np.float32)).to(
                            #     geodataloader.device
                            # )
                            geoGrid = geoGrid.to(
                                geodataloader.device, non_blocking=async_transfer
                            )
                            ygeo = ygeo.to(
                                geodataloader.device, non_blocking=async_transfer
                            )
                            errgeo = errgeo.to(
                                geodataloader.device, non_blocking=async_transfer
                            )
                            geod_periods = geodataloader.periods_per_glacier[current_g]
                            lossGeo += compute_geo_loss(
                                model,
                                geoGrid,
                                metadata,
                                ygeo,
                                errgeo,
                                geod_periods,
                                precomputed_meta,
                            )
                            cntGeo += 1

                        # Shift pipeline
                        current_g = next_g
                        current_geo_future = next_geo_future
                        batch_idx += 1

                        cntStake += valScalingStakes
                    lossStake /= cntStake
                    if wGeo > 0 and cntGeo > 0:
                        lossGeo /= cntGeo
                        loss = lossStake + wGeo * lossGeo
                    else:
                        lossGeo = torch.tensor(torch.nan)
                        loss = lossStake
                    mse, rmse, mae, pearson_corr, r2, bias = scores(predAll, targetAll)

                    indAnnual = np.argwhere(periodAll == "annual")[:, 0]
                    indWinter = np.argwhere(periodAll == "winter")[:, 0]
                    predAnnual = predAll[indAnnual]
                    targetAnnual = targetAll[indAnnual]
                    predWinter = predAll[indWinter]
                    targetWinter = targetAll[indWinter]
                    (
                        mse_annual,
                        rmse_annual,
                        mae_annual,
                        pearson_corr_annual,
                        r2_annual,
                        bias_annual,
                    ) = scores(predAnnual, targetAnnual)
                    (
                        mse_winter,
                        rmse_winter,
                        mae_winter,
                        pearson_corr_winter,
                        r2_winter,
                        bias_winter,
                    ) = scores(predWinter, targetWinter)

                    statsVal["lossValStake"].append(lossStake.item())
                    statsVal["lossValGeo"].append(lossGeo.item())
                    statsVal["lossVal"].append(loss.item())
                    statsVal["mse"].append(mse.item())
                    statsVal["rmse"].append(rmse.item())
                    statsVal["mae"].append(mae.item())
                    statsVal["pearson"].append(pearson_corr.item())
                    statsVal["r2"].append(r2.item())
                    statsVal["bias"].append(bias.item())
                    statsVal["mse_annual"].append(mse_annual.item())
                    statsVal["rmse_annual"].append(rmse_annual.item())
                    statsVal["mae_annual"].append(mae_annual.item())
                    statsVal["pearson_annual"].append(pearson_corr_annual.item())
                    statsVal["r2_annual"].append(r2_annual.item())
                    statsVal["bias_annual"].append(bias_annual.item())
                    statsVal["mse_winter"].append(mse_winter.item())
                    statsVal["rmse_winter"].append(rmse_winter.item())
                    statsVal["mae_winter"].append(mae_winter.item())
                    statsVal["pearson_winter"].append(pearson_corr_winter.item())
                    statsVal["r2_winter"].append(r2_winter.item())
                    statsVal["bias_winter"].append(bias_winter.item())

                    # Log to TensorBoard
                    writer.add_scalar("LossStake/val", lossStake.item(), epoch)
                    writer.add_scalar("LossGeo/val", lossGeo.item(), epoch)
                    writer.add_scalar("Loss/val", loss.item(), epoch)
                    writer.add_scalar("RMSE/val", rmse.item(), epoch)
                    writer.add_scalar("RMSE_annual/val", rmse_annual.item(), epoch)
                    writer.add_scalar("RMSE_winter/val", rmse_winter.item(), epoch)
                    writer.add_scalar("MAE/val", mae.item(), epoch)
                    writer.add_scalar("MAE_annual/val", mae_annual.item(), epoch)
                    writer.add_scalar("MAE_winter/val", mae_winter.item(), epoch)
                    writer.add_scalar("Pearson/val", pearson_corr.item(), epoch)
                    writer.add_scalar(
                        "Pearson_annual/val", pearson_corr_annual.item(), epoch
                    )
                    writer.add_scalar(
                        "Pearson_winter/val", pearson_corr_winter.item(), epoch
                    )
                    writer.add_scalar("R2/val", r2.item(), epoch)
                    writer.add_scalar("R2_annual/val", r2_annual.item(), epoch)
                    writer.add_scalar("R2_winter/val", r2_winter.item(), epoch)
                    writer.add_scalar("bias/val", bias.item(), epoch)
                    writer.add_scalar("bias_annual/val", bias_annual.item(), epoch)
                    writer.add_scalar("bias_winter/val", bias_winter.item(), epoch)

                    # Check if current model is among the top 5
                    scalarBestModelCriterion = (
                        statsVal[bestModelCriterion][-1]
                        if len(statsVal[bestModelCriterion]) > 0
                        else np.nan
                    )
                    assert not np.isnan(
                        scalarBestModelCriterion
                    ), f"The statistics {bestModelCriterion} used for best model selection contains NaN."
                    higherBetterCriterion = (
                        1
                        if any(crit in bestModelCriterion for crit in _maxCriterion)
                        else -1
                    )
                    model_path = os.path.join(
                        log_dir,
                        f"model_epoch{epoch}_{bestModelCriterion}{scalarBestModelCriterion:.4f}.pt",
                    )

                    if (
                        len(top_models) < 5
                        or scalarBestModelCriterion
                        < higherBetterCriterion * top_models[0][0]
                    ):

                        if len(top_models) >= 5:
                            # Pop the worst among top 5
                            _, worst_path = heapq.heappop(top_models)
                            if os.path.exists(worst_path):
                                os.remove(worst_path)

                        # Save the new model
                        # Use -scalarBestModelCriterion for max-heap
                        heapq.heappush(
                            top_models,
                            (
                                higherBetterCriterion * scalarBestModelCriterion,
                                model_path,
                            ),
                        )
                        torch.save(model.state_dict(), model_path)

                        if (
                            geodataloader_test is not None
                            and len(geodataloader_test) > 0
                        ):
                            assessOnTest(log_dir, model, geodataloader_test, light=True)

            rmse = statsVal["rmse"][-1] if len(statsVal["rmse"]) > 0 else np.nan
            pearson = (
                statsVal["pearson"][-1] if len(statsVal["pearson"]) > 0 else np.nan
            )
            loss = statsVal["lossVal"][-1] if len(statsVal["lossVal"]) > 0 else np.nan

            geodataloader.onEpochEnd()

            # Show progress bar
            tqdm.tqdm.write(
                f"[Epoch {epoch+1} / {Nepochs}] Avg training loss: {avg_training_loss:.3f}, Val loss: {loss:.3f}, RMSE: {rmse:.3f}, Pearson: {pearson:.3f}"
            )

    except KeyboardInterrupt:
        print("Stopping training after KeyboardInterrupt...")

    # Close TensorBoard
    writer.close()

    return {
        "training": statsTraining,
        "validation": statsVal,
        "misc": {"log_dir": log_dir},
    }

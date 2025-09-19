import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import r2_score
import numpy as np
import datetime
import heapq
import matplotlib.pyplot as plt
import pandas as pd
import glob
import warnings

from plots import predVSTruth


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
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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
        stakes (nd.ndarray): Stake measurements features returned by the geodetic
            dataloader.
        metadata (pd.DataFrame): Metadata returned by the geodetic dataloader.
        point_balance (np.ndarray): Mass balance value of each stake measurement.
            It is returned by the geodetic dataloader.
        returnPred (bool): Whether to return the prediction and the target in a
            dictionary. Default is False.

    Returns a scalar torch value that corresponds to the stake loss term and
    optionally statistics in a dictionary.
    """
    idAggr = metadata["ID_int"].values

    # Make prediction
    stakesTorch = torch.tensor(stakes.astype(np.float32))
    pred = model.forward(stakesTorch)[:, 0]

    # Aggregate per stake and periods
    groundTruthTorch = torch.tensor(point_balance.astype(np.float32))
    trueMean = model.aggrPredict(groundTruthTorch, idAggr, reduce="mean")
    predSum = model.aggrPredict(pred, idAggr)

    mse = nn.functional.mse_loss(predSum, trueMean, reduction="mean")
    ret = {}
    if returnPred:
        ret["target"] = trueMean.detach()
        ret["pred"] = predSum.detach()
    return mse, ret


def timeWindowGeodeticLoss(
    predSumAnnualGlwd, geoTarget, metadataAggrYear, geod_periods
):
    """
    Given glacier wide mass balance values for different years, this function
    computes the predicted geodetic MB values over different time windows and then
    computes the loss term for each of these windows.

    Args:
        predSumAnnualGlwd (torch.Tensor): Predicted MB values for different years.
        geoTarget (torch.Tensor): Ground truth MB values for the different time windows.
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

    geodetic_MB_pred = torch.zeros(len(geod_periods))
    for e, (start_year, end_year) in enumerate(geod_periods):
        geodetic_range = range(start_year, end_year + 1)

        # Ensure years exist in index before selection
        valid_years = [yr for yr in geodetic_range if yr in yearsPred]
        valid_years = []
        indSlice = []
        for yr in geodetic_range:
            if yr in yearsPred:
                valid_years.append(yr)
                indSlice.append(np.argwhere(yearsPred == yr)[0, 0])
        if valid_years:
            geodetic_MB_pred[e] = (
                torch.mean(predSumAnnualGlwd[indSlice]) - geoTarget[e]
            ) ** 2
        else:
            geodetic_MB_pred[e] = np.nan  # Handle missing years
    return geodetic_MB_pred


def compute_geo_loss(model, geoGrid, metadata, ygeo, geod_periods):
    """
    Computes the geodetic loss term.

    Args:
        model (CustomTorchNeuralNetRegressor): Model to evaluate.
        geoGrid (nd.ndarray): Geodetic features returned by the geodetic dataloader.
        metadata (pd.DataFrame): Metadata returned by the geodetic dataloader.
        ygeo (np.ndarray): Geodetic mass balance values. It is returned by the
            geodetic dataloader.

    Returns a scalar torch value that corresponds to the geodetic loss term.
    """
    # Make prediction
    geoGridTorch = torch.tensor(geoGrid.astype(np.float32))
    pred = model.forward(geoGridTorch)[:, 0]

    # Aggregate per point on the grid
    grouped_ids = model.aggrMetadataId(metadata, "ID_int")
    predSumAnnual = model.aggrPredict(pred, metadata["ID_int"].values)

    # Aggregate glacier wide
    metadataAggrYear = model.aggrMetadataGlwdId(grouped_ids, "GLWD_ID_int")
    predSumAnnualGlwd = model.aggrPredictGlwd(
        predSumAnnual, grouped_ids["GLWD_ID_int"].values
    )

    groundTruthTorch = torch.tensor(ygeo.astype(np.float32))

    # Compute the geodetic MB for the different time windows
    lossGeo = timeWindowGeodeticLoss(
        predSumAnnualGlwd, groundTruthTorch, metadataAggrYear, geod_periods
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


def assessOnTest(log_dir, model, geodataloader_test):
    targetAll = torch.zeros(0)
    predAll = torch.zeros(0)
    periodAll = torch.zeros(0, dtype=torch.int)
    for g in geodataloader_test.glaciers():
        stakes, metadata, point_balance = geodataloader_test.stakes(g)
        l, ret = compute_stake_loss(
            model, stakes, metadata, point_balance, returnPred=True
        )
        targetAll = torch.concatenate((targetAll, ret["target"]))
        predAll = torch.concatenate((predAll, ret["pred"]))
        grouped_ids = metadata.groupby("ID_int").agg({"PERIOD_int": "first"})
        periodAll = torch.concatenate(
            (periodAll, torch.tensor(grouped_ids["PERIOD_int"].values))
        )
    mse, rmse, mae, pearson_corr, r2, bias = scores(predAll, targetAll)

    indAnnual = torch.argwhere(periodAll == geodataloader_test.periodToInt["annual"])[
        :, 0
    ]
    indWinter = torch.argwhere(periodAll == geodataloader_test.periodToInt["winter"])[
        :, 0
    ]
    predAnnual = predAll[indAnnual]
    targetAnnual = targetAll[indAnnual]
    predWinter = predAll[indWinter]
    targetWinter = targetAll[indWinter]
    mse_annual, rmse_annual, mae_annual, pearson_corr_annual, r2_annual, bias_annual = (
        scores(predAnnual, targetAnnual)
    )
    mse_winter, rmse_winter, mae_winter, pearson_corr_winter, r2_winter, bias_winter = (
        scores(predWinter, targetWinter)
    )

    # Make plots
    plot_pred_vs_obs(log_dir, targetAll, predAll, {"rmse": rmse, "mae": mae})

    return {
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
    model.load_state_dict(torch.load(files[best], weights_only=True))
    return files[best]


def train_geo(
    model,
    geodataloader,
    optim,
    trainCfg,
    scheduler=None,
    geodataloader_test=None,
):
    """
    Train a model with both stake measurements and geodetic data.

    Args:
        model (CustomTorchNeuralNetRegressor): Model to train.
        geodataloader (GeoDataLoader): Dataloader that provides both stake
            measurements and geodetic data.
        optim (PyTorch optimizer): Optimizer instance to use.
        trainCfg (dict): Trainin options.
        scheduler (PyTorch LR scheduler): The learning rate scheduler (optional).
        geodataloader_test (GeoDataLoader): Optional dataloader that provides both
            stake measurements and geodetic data on the test set.
    """
    Nepochs = trainCfg["Nepochs"]
    wGeo = trainCfg.get("wGeo", 1.0)
    freqVal = trainCfg.get("freqVal", 1)
    bestModelCriterion = trainCfg.get("bestModelCriterion", "lossVal")
    assert bestModelCriterion in _criterionVal
    scalingStakes = trainCfg.get("scalingStakes", "glacier")
    assert scalingStakes in ["meas", "glacier"]
    iterPerEpoch = len(geodataloader)
    nColsProgressBar = 500 if _inJupyterNotebook else 100

    # Setup logging
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_suffix = trainCfg.get("log_suffix", "")
    if log_suffix != "":
        log_suffix = "_" + log_suffix
    log_dir = trainCfg.get(
        "log_dir",
        os.path.join(
            _default_log_dir,
            f"geo_{run_name}{log_suffix}",
        ),
    )
    os.makedirs(log_dir, exist_ok=True)

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
    for suffix in ["", "_annual", "_winter"]:
        for metric in valMetrics:
            statsVal[metric + suffix] = []
    top_models = []  # Heap of (val_loss, filepath) to store top 5 models

    for epoch in tqdm(
        range(Nepochs), desc="Epochs", position=0, leave=True, ncols=nColsProgressBar
    ):

        avg_training_loss = 0.0
        model.train()
        with tqdm(
            total=iterPerEpoch,
            desc=f"Batch",
            position=1,
            leave=False,
            ncols=nColsProgressBar,
        ) as batch_bar:
            for batch_idx, g in enumerate(geodataloader.glaciers()):
                optim.zero_grad()

                stakes, metadata, point_balance = geodataloader.stakes(g)
                lossStake, _ = compute_stake_loss(
                    model, stakes, metadata, point_balance
                )

                valScalingStakes = stakes.shape[0] if scalingStakes == "meas" else 1.0
                lossStake = lossStake * valScalingStakes
                if wGeo > 0 and geodataloader.hasGeo(g):
                    geoGrid, metadata, ygeo = geodataloader.geo(g)
                    geod_periods = geodataloader.periods_per_glacier[g]
                    lossGeo = compute_geo_loss(
                        model, geoGrid, metadata, ygeo, geod_periods
                    )

                    loss = lossStake + wGeo * lossGeo
                else:
                    lossGeo = torch.tensor(torch.nan)
                    loss = lossStake

                loss.backward()
                optim.step()

                lr = optim.param_groups[0]["lr"]
                statsTraining["lossStake"].append(lossStake.item())
                statsTraining["lossGeo"].append(lossGeo.item())
                statsTraining["loss"].append(loss.item())
                statsTraining["lr"].append(lr)
                avg_training_loss += statsTraining["loss"][-1]

                # Log to TensorBoard
                globalStep = iterPerEpoch * epoch + batch_idx
                writer.add_scalar("Loss/train", loss.item(), globalStep)
                writer.add_scalar("LossStake/train", lossStake.item(), globalStep)
                writer.add_scalar("LossGeo/train", lossGeo.item(), globalStep)
                writer.add_scalar("Step", optim.param_groups[0]["lr"], globalStep)

                batch_bar.set_postfix(
                    loss=f"{statsTraining['loss'][-1]:.4f}",
                    lossStake=f"{statsTraining['lossStake'][-1]:.4f}",
                    lossGeo=f"{statsTraining['lossGeo'][-1]:.4f}",
                )
                batch_bar.update(1)

        avg_training_loss /= iterPerEpoch
        if scheduler is not None:
            scheduler.step()

        if freqVal:
            model.eval()
            with torch.no_grad():
                cntStake = 0
                cntGeo = 0
                lossStake = 0.0
                lossGeo = 0.0
                targetAll = torch.zeros(0)
                predAll = torch.zeros(0)
                periodAll = torch.zeros(0, dtype=torch.int)
                for g in geodataloader.glaciers():

                    stakes, metadata, point_balance = geodataloader.stakes(g)
                    l, ret = compute_stake_loss(
                        model, stakes, metadata, point_balance, returnPred=True
                    )
                    target = ret["target"]
                    pred = ret["pred"]
                    targetAll = torch.concatenate((targetAll, target))
                    predAll = torch.concatenate((predAll, pred))
                    grouped_ids = metadata.groupby("ID_int").agg(
                        {"PERIOD_int": "first"}
                    )
                    periodAll = torch.concatenate(
                        (periodAll, torch.tensor(grouped_ids["PERIOD_int"].values))
                    )

                    valScalingStakes = (
                        stakes.shape[0] if scalingStakes == "meas" else 1.0
                    )
                    l = l * valScalingStakes

                    lossStake += l

                    if wGeo > 0 and geodataloader.hasGeo(g):
                        geoGrid, metadata, ygeo = geodataloader.geo(g)
                        geod_periods = geodataloader.periods_per_glacier[g]
                        lossGeo += compute_geo_loss(
                            model, geoGrid, metadata, ygeo, geod_periods
                        )
                        cntGeo += 1

                    cntStake += valScalingStakes
                lossStake /= cntStake
                if wGeo > 0 and cntGeo > 0:
                    lossGeo /= cntGeo
                    loss = lossStake + wGeo * lossGeo
                else:
                    lossGeo = torch.tensor(torch.nan)
                    loss = lossStake
                mse, rmse, mae, pearson_corr, r2, bias = scores(predAll, targetAll)

                indAnnual = torch.argwhere(
                    periodAll == geodataloader.periodToInt["annual"]
                )[:, 0]
                indWinter = torch.argwhere(
                    periodAll == geodataloader.periodToInt["winter"]
                )[:, 0]
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
                        (higherBetterCriterion * scalarBestModelCriterion, model_path),
                    )
                    torch.save(model.state_dict(), model_path)

                    if geodataloader_test is not None:
                        assessOnTest(log_dir, model, geodataloader_test)

        rmse = statsVal["rmse"][-1] if len(statsVal["rmse"]) > 0 else np.nan
        pearson = statsVal["pearson"][-1] if len(statsVal["pearson"]) > 0 else np.nan
        loss = statsVal["lossVal"][-1] if len(statsVal["lossVal"]) > 0 else np.nan

        # Show progress bar
        tqdm.write(
            f"[Epoch {epoch+1} / {Nepochs}] Avg training loss: {avg_training_loss:.3f}, Val loss: {loss:.3f}, RMSE: {rmse:.3f}, Pearson: {pearson:.3f}"
        )

    # Close TensorBoard
    writer.close()

    return {
        "training": statsTraining,
        "validation": statsVal,
        "misc": {"log_dir": log_dir},
    }

import torch
from torch import nn
import numpy as np

def in_jupyter_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        pass
    return False

_inJupyterNotebook = in_jupyter_notebook()
if _inJupyterNotebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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
    idAggr = metadata['ID_int'].values

    # Make prediction
    stakesTorch = torch.tensor(stakes.astype(np.float32))
    pred = model.forward(stakesTorch)[:, 0]

    # Aggregate per stake and periods
    groundTruthTorch = torch.tensor(point_balance.astype(np.float32))
    trueMean = model.aggrPredict(groundTruthTorch, idAggr, reduce='mean')
    predSum = model.aggrPredict(pred, idAggr)

    mse = nn.functional.mse_loss(predSum, trueMean, reduction='mean')
    ret = {}
    if returnPred:
        ret['target'] = trueMean.detach()
        ret['pred'] = predSum.detach()
    return mse, ret

def timeWindowGeodeticLoss(predSumAnnualGlwd, geoTarget, metadataAggrYear, geod_periods):
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
    assert len(geoTarget) == len(geod_periods), f"Size of the ground truth is {geoTarget.shape} but doesn't match with the number of geodetic periods which is {len(geod_periods)}"
    yearsPred = metadataAggrYear.YEAR.values

    geodetic_MB_pred = torch.zeros(len(geod_periods))
    for e, (start_year, end_year) in enumerate(geod_periods):
        geodetic_range = range(start_year, end_year + 1)

        # Ensure years exist in index before selection
        valid_years = [
            yr for yr in geodetic_range if yr in yearsPred
        ]
        valid_years = []
        indSlice = []
        for yr in geodetic_range:
            if yr in yearsPred:
                valid_years.append(yr)
                indSlice.append(np.argwhere(yearsPred==yr)[0,0])
        if valid_years:
            geodetic_MB_pred[e] = (torch.mean(predSumAnnualGlwd[indSlice]) - geoTarget[e])**2
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
    grouped_ids = model.aggrMetadataId(metadata, 'ID_int')
    predSumAnnual = model.aggrPredict(pred, metadata['ID_int'].values)

    # Aggregate glacier wide
    metadataAggrYear = model.aggrMetadataGlwdId(grouped_ids, 'GLWD_ID_int')
    predSumAnnualGlwd = model.aggrPredictGlwd(predSumAnnual, grouped_ids['GLWD_ID_int'].values)

    groundTruthTorch = torch.tensor(ygeo.astype(np.float32))

    # Compute the geodetic MB for the different time windows
    lossGeo = timeWindowGeodeticLoss(predSumAnnualGlwd, groundTruthTorch, metadataAggrYear, geod_periods)

    return lossGeo.mean() # Compute mean of the different time window scores


def computeScores(pred:torch.Tensor, target:torch.Tensor):
    mse = nn.functional.mse_loss(pred, target, reduction='mean')
    rmse = mse.sqrt()
    pearson_corr = torch.corrcoef(torch.cat((pred[:,None], target[:,None]), axis=1).T)[0, 1]
    return mse, rmse, pearson_corr


def train_geo(model, geodataloader, optim, trainCfg, scheduler=None):
    """
    Train a model with both stake measurements and geodetic data.

    Args:
        model (CustomTorchNeuralNetRegressor): Model to train.
        geodataloader (GeoDataLoader): Dataloader that provides both stake
            measurements and geodetic data.
        optim (PyTorch optimizer): Optimizer instance to use.
        trainCfg (dict): Trainin options.
        scheduler (PyTorch LR scheduler): The learning rate scheduler (optional).
    """
    Nepochs = trainCfg['Nepochs']
    wGeo = trainCfg.get('wGeo', 1.0)
    freqVal = trainCfg.get('freqVal', 1)
    iterPerEpoch = len(geodataloader)
    nColsProgressBar = 500 if _inJupyterNotebook else 100

    statsTraining = {
        'loss': [],
        'lossStake': [],
        'lossGeo': [],
        'lr': [],
    }
    statsVal = {
        'lossVal': [],
        'lossValStake': [],
        'lossValGeo': [],
        'mse': [],
        'rmse': [],
        'pearson': [],
        'mse_annual': [],
        'rmse_annual': [],
        'pearson_annual': [],
        'mse_winter': [],
        'rmse_winter': [],
        'pearson_winter': [],
    }
    for epoch in tqdm(range(Nepochs), desc="Epochs", position=0, leave=True, ncols=nColsProgressBar):

        avg_training_loss = 0.
        model.train()
        with tqdm(total=iterPerEpoch, desc=f"Batch", position=1, leave=False, ncols=nColsProgressBar) as batch_bar:
            for g in geodataloader.glaciers():
                optim.zero_grad()

                stakes, metadata, point_balance = geodataloader.stakes(g)
                lossStake, _ = compute_stake_loss(model, stakes, metadata, point_balance)

                if wGeo > 0 and geodataloader.hasGeo(g):
                    geoGrid, metadata, ygeo = geodataloader.geo(g)
                    geod_periods = geodataloader.periods_per_glacier[g]
                    lossGeo = compute_geo_loss(model, geoGrid, metadata, ygeo, geod_periods)

                    loss = lossStake + wGeo * lossGeo
                else:
                    lossGeo = torch.tensor(torch.nan)
                    loss = lossStake

                loss.backward()
                optim.step()

                lr = optim.param_groups[0]["lr"]
                statsTraining['lossStake'].append(lossStake.item())
                statsTraining['lossGeo'].append(lossGeo.item())
                statsTraining['loss'].append(loss.item())
                statsTraining['lr'].append(lr)
                avg_training_loss += statsTraining['loss'][-1]
                batch_bar.set_postfix(loss=f"{statsTraining['loss'][-1]:.4f}", lossStake=f"{statsTraining['lossStake'][-1]:.4f}", lossGeo=f"{statsTraining['lossGeo'][-1]:.4f}")
                batch_bar.update(1)

        avg_training_loss /= iterPerEpoch
        if scheduler is not None:
            scheduler.step()

        if freqVal:
            model.eval()
            with torch.no_grad():
                cntStake = 0
                cntGeo = 0
                lossStake = 0.
                lossGeo = 0.
                targetAll = torch.zeros(0)
                predAll = torch.zeros(0)
                periodAll = torch.zeros(0, dtype=torch.int)
                for g in geodataloader.glaciers():

                    stakes, metadata, point_balance = geodataloader.stakes(g)
                    l, ret = compute_stake_loss(model, stakes, metadata, point_balance, returnPred=True)
                    target = ret['target']
                    pred = ret['pred']
                    targetAll = torch.concatenate((targetAll, target))
                    predAll = torch.concatenate((predAll, pred))
                    grouped_ids = metadata.groupby('ID_int').agg({'PERIOD_int': 'first'})
                    periodAll = torch.concatenate((periodAll, torch.tensor(grouped_ids['PERIOD_int'].values)))
                    lossStake += l

                    if wGeo > 0 and geodataloader.hasGeo(g):
                        geoGrid, metadata, ygeo = geodataloader.geo(g)
                        geod_periods = geodataloader.periods_per_glacier[g]
                        lossGeo += compute_geo_loss(model, geoGrid, metadata, ygeo, geod_periods)
                        cntGeo += 1

                    cntStake += 1
                lossStake /= cntStake
                if wGeo > 0 and cntGeo > 0:
                    lossGeo /= cntGeo
                    loss = lossStake + wGeo * lossGeo
                else:
                    lossGeo = torch.tensor(torch.nan)
                    loss = lossStake
                mse, rmse, pearson_corr = computeScores(predAll, targetAll)

                indAnnual = torch.argwhere(periodAll==geodataloader.periodToInt['annual'])[:,0]
                indWinter = torch.argwhere(periodAll==geodataloader.periodToInt['winter'])[:,0]
                print()
                print(f"{predAll.shape=}")
                print(f"{periodAll.shape=}")
                print(f"{targetAll.shape=}")
                print()
                predAnnual = predAll[indAnnual]
                targetAnnual = targetAll[indAnnual]
                predWinter = predAll[indWinter]
                targetWinter = targetAll[indWinter]
                mse_annual, rmse_annual, pearson_corr_annual = computeScores(predAnnual, targetAnnual)
                mse_winter, rmse_winter, pearson_corr_winter = computeScores(predWinter, targetWinter)

                statsVal['lossValStake'].append(lossStake.item())
                statsVal['lossValGeo'].append(lossGeo.item())
                statsVal['lossVal'].append(loss.item())
                statsVal['mse'].append(mse.item())
                statsVal['rmse'].append(rmse.item())
                statsVal['pearson'].append(pearson_corr.item())
                statsVal['mse_annual'].append(mse_annual.item())
                statsVal['rmse_annual'].append(rmse_annual.item())
                statsVal['pearson_annual'].append(pearson_corr_annual.item())
                statsVal['mse_winter'].append(mse_winter.item())
                statsVal['rmse_winter'].append(rmse_winter.item())
                statsVal['pearson_winter'].append(pearson_corr_winter.item())

        mse = statsVal['mse'][-1] if len(statsVal['mse'])>0 else np.nan
        pearson = statsVal['pearson'][-1] if len(statsVal['pearson'])>0 else np.nan
        loss = statsVal['lossVal'][-1] if len(statsVal['lossVal'])>0 else np.nan
        tqdm.write(f"[Epoch {epoch+1} / {Nepochs}] Avg training loss: {avg_training_loss:.3f}, Val loss: {loss:.3f}, MSE: {mse:.3f}, Pearson: {pearson:.3f}")

    return {'training': statsTraining, 'validation': statsVal}

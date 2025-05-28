import torch
from torch import nn
import numpy as np


def compute_stake_loss(model, stakes, metadata, point_balance):
    """
    Computes the stake loss term.

    Args:
        model (CustomTorchNeuralNetRegressor): Model to evaluate.
        stakes (nd.ndarray): Stake measurements features returned by the geodetic
            dataloader.
        metadata (pd.DataFrame): Metadata returned by the geodetic dataloader.
        point_balance (np.ndarray): Mass balance value of each stake measurement.
            It is returned by the geodetic dataloader.

    Returns a scalar torch value that corresponds to the stake loss term.
    """
    assert model.training, "This function is not designed to be used in eval mode."
    idAggr = metadata['ID_int'].values

    # Make prediction
    stakesTorch = torch.tensor(stakes.astype(np.float32))
    pred = model.forward(stakesTorch)[:, 0]

    # Aggregate per stake and periods
    groundTruthTorch = torch.tensor(point_balance.astype(np.float32))
    trueMean = model.aggrPredict(groundTruthTorch, idAggr, reduce='mean')
    predSum = model.aggrPredict(pred, idAggr)

    lossStake = nn.functional.mse_loss(predSum, trueMean, reduce='mean')
    return lossStake

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
            geodetic_MB_pred[e] = torch.mean(predSumAnnualGlwd[indSlice]) - geoTarget[e]
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
    assert model.training, "This function is not designed to be used in eval mode."
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


def train_geo(model, geodataloader, optim, trainCfg):
    """
    Train a model with both stake measurements and geodetic data.

    Args:
        model (CustomTorchNeuralNetRegressor): Model to train.
        geodataloader (GeoDataLoader): Dataloader that provides both stake
            measurements and geodetic data.
        optim (PyTorch optimizer): Optimizer instance to use.
        trainCfg (dict): Trainin options.
    """
    Nepochs = trainCfg['Nepochs']
    wGeo = trainCfg.get('wGeo', 1.0)

    model.train()
    lossHist = []
    lossStakeHist = []
    lossGeoHist = []
    for epoch in range(Nepochs):
        for g in geodataloader.glaciers():
            optim.zero_grad()

            stakes, metadata, point_balance = geodataloader.stakes(g)
            lossStake = compute_stake_loss(model, stakes, metadata, point_balance)

            if wGeo > 0:
                geoGrid, metadata, ygeo = geodataloader.geo(g)
                geod_periods = geodataloader.periods_per_glacier[g]
                lossGeo = compute_geo_loss(model, geoGrid, metadata, ygeo, geod_periods)

                loss = lossStake + wGeo * lossGeo
            else:
                loss = lossStake

            loss.backward()
            optim.step()

            lossStakeHist.append(lossStake.item())
            lossGeoHist.append(lossGeo.item())
            lossHist.append(loss.item())
            print(f"{epoch}  loss = {lossHist[-1]}")

    return {
        'lossHist': lossHist,
        'lossStakeHist': lossStakeHist,
        'lossGeoHist': lossGeoHist,
    }

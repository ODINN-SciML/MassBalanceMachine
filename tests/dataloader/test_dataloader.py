from collections import deque

import pandas as pd
import massbalancemachine as mbm

from massbalancemachine.training.training import _prefetch_geo_batches


class DummyGeoLoader:
    def __init__(self):
        self.submitted = []

    def hasGeo(self, glacier_name):
        return True

    def submit_geo(self, glacier_name):
        self.submitted.append(glacier_name)
        return glacier_name


def test_prefetch_geo_batches_queue_multiple_items():
    loader = DummyGeoLoader()
    glacier_iter = iter(["g1", "g2", "g3"])

    queue = _prefetch_geo_batches(glacier_iter, loader, wGeo=1, prefetch_batches=2)

    assert isinstance(queue, deque)
    assert list(queue) == [("g1", "g1"), ("g2", "g2")]
    assert loader.submitted == ["g1", "g2"]


def test_prefetch_geo_batches_skips_when_wgeo_zero():
    loader = DummyGeoLoader()
    glacier_iter = iter(["g1", "g2"])

    queue = _prefetch_geo_batches(glacier_iter, loader, wGeo=0, prefetch_batches=2)

    assert queue == deque()
    assert loader.submitted == []


def test_dataloader():
    data = pd.read_csv(
        "./notebooks/example_data/iceland/files/iceland_monthly_dataset.csv"
    )

    cfg = mbm.Config(seed=30)

    # Create a new DataLoader object with the monthly stake data measurements
    dataloader = mbm.dataloader.DataLoader(cfg, data=data, random_seed=0)

    # Test both kfold types
    train_itr, test_itr = dataloader.set_train_test_split(
        test_size=0.3, type_fold="group-rgi"
    )
    train_itr, test_itr = dataloader.set_train_test_split(
        test_size=0.3, type_fold="group-meas-id"
    )

    # Get all indices of the training and testing dataset at once from the iterators
    train_indices, test_indices = list(train_itr), list(test_itr)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation
    df_X_train = data.iloc[train_indices]
    y_train = df_X_train["POINT_BALANCE"].values
    assert df_X_train.shape == (305, 21)
    assert y_train.shape == (305,)

    df_X_test = data.iloc[test_indices]
    y_test = df_X_test["POINT_BALANCE"].values
    assert df_X_test.shape == (140, 21)
    assert y_test.shape == (140,)

    # Create the cross validation splits based on the training dataset
    splits = dataloader.get_cv_split(n_splits=5, type_fold="group-meas-id")
    assert len(splits) == 5
    for v in splits:
        assert len(v) == 2


if __name__ == "__main__":
    test_dataloader()

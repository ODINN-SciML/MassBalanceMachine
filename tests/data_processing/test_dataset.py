import numpy as np
import massbalancemachine as mbm


def test_aggregatedDataset():

    def check_vector_equality(var, ref, name):
        assert var.shape==ref.shape, f"{name} doesn't have the same shape as target. Got {var.shape} but expected {ref.shape}"
        assert ((var == ref) | (np.isnan(var) & np.isnan(ref))).all(), f"{name} isn't equal to the reference. Got {var} but expected {ref}"

    features = np.array([
        [1., 2.],
        [3., 4.],
        [5., 6.],
    ])
    metadata = np.array([[1, 'sep'], [1, 'oct'], [2, 'jan']])
    meta_data_columns = ['ID', 'MONTHS']
    targets = np.array([10, 11, 12])
    splits = [(np.array([0, 1]), np.array([2]))]

    cfg = mbm.Config(bnds={'COL1': (0., 10.), 'COL2': (-1., 1.0)}, seed=30)

    cfg.setFeatures(['COL1', 'COL2'])

    dataset = mbm.data_processing.AggregatedDataset(cfg, features=features, metadata=metadata, metadataColumns=meta_data_columns, targets=targets)
    assert len(dataset)==2

    splits = dataset.mapSplitsToDataset(splits)
    assert splits==[(np.array([0]),np.array([1]))]

    features0 = dataset[0][0]
    target0 = dataset[0][1]
    check_vector_equality(features0, np.array([0.1,1.5,0.3,2.5]), "features0")
    check_vector_equality(target0, np.array([10.,11.]), "target0")
    features1 = dataset[1][0]
    target1 = dataset[1][1]
    check_vector_equality(features1, np.array([0.5,3.5,np.nan,np.nan]), "features1")
    check_vector_equality(target1, np.array([12.,np.nan]), "target1")
    assert (dataset.indexToMetadata(0)==[['1', 'sep'], ['1', 'oct']]).all()


if __name__=="__main__":
    test_aggregatedDataset()
import pandas as pd
import massbalancemachine as mbm


def test_dataloader():
    data = pd.read_csv('./notebooks/example_data/iceland/files/iceland_monthly_dataset.csv')

    cfg = mbm.Config()

    # Create a new DataLoader object with the monthly stake data measurements
    dataloader = mbm.dataloader.DataLoader(cfg, data=data, random_seed=0)

    # Test both kfold types
    train_itr, test_itr = dataloader.set_train_test_split(test_size=0.3, type_fold='group-rgi')
    train_itr, test_itr = dataloader.set_train_test_split(test_size=0.3, type_fold='group-meas-id')

    # Get all indices of the training and testing dataset at once from the iterators
    train_indices, test_indices = list(train_itr), list(test_itr)

    # Get the features and targets of the training data for the indices as defined above, that will be used during the cross validation
    df_X_train = data.iloc[train_indices]
    y_train = df_X_train['POINT_BALANCE'].values
    assert df_X_train.shape == (296,20)
    assert y_train.shape == (296,)

    df_X_test = data.iloc[test_indices]
    y_test = df_X_test['POINT_BALANCE'].values
    assert df_X_test.shape == (151,20)
    assert y_test.shape == (151,)

    # Create the cross validation splits based on the training dataset
    splits = dataloader.get_cv_split(n_splits=5, type_fold='group-meas-id')
    assert len(splits)==5
    for v in splits:
        assert len(v)==2


if __name__=="__main__":
    test_dataloader()

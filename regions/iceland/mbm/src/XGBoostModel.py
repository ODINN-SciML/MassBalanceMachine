class XGBoostModel:

    def __init__(self, x_train, y_train, cv_splits, params):
        self.X_train = x_train
        self.y_train = y_train
        self.cv_splits = cv_splits
        self.params = params

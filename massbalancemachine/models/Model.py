from massbalancemachine.dataloader import DataLoader


class Model():
    def __init__(self):
        pass

    def fitt(self, dataloader: DataLoader, loss='reg:squarederror'):
        pass

    def predict(self, dataloader: DataLoader):
        pass

    def perform_gridsearch(self, dataloader: DataLoader, random=True, loss='reg:squarederror', score='reg:squarederror', **params):
        pass

    def monthly_loss(self, metric='MSE'):
        pass

    def score(self):
        pass
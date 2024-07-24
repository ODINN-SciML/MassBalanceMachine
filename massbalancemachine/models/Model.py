from abc import ABC, abstractmethod
from massbalancemachine.dataloader import DataLoader


class Model(ABC):
    def __init__(self): #**params
        pass
        # # TODO
        # default_params = {
        #     ...
        # }
        # default_params.update(params)
        # self.params = default_params

    @abstractmethod
    def fit(self, dataloader: DataLoader, loss='reg:squarederror'):
        pass

    @abstractmethod
    def predict(self, dataloader: DataLoader):
        pass

    @abstractmethod
    def perform_gridsearch(self, dataloader: DataLoader, random=True, loss='reg:squarederror', score='reg:squarederror', **params):
        pass

    @abstractmethod
    def monthly_loss(self, metric='MSE'):
        pass

    @abstractmethod
    def score(self):
        pass
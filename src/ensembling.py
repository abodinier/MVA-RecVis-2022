import torch
import numpy as np


class Averaging:
    def __init__(self, models: list, weights: torch.tensor = None) -> None:
        self.models = models
        self.n_models = len(models)
        self.nclasses = models[0].nclasses
        if weights is not None:
            assert len(weights) == len(models), f"There are {len(models)} models and {len(weights)} weights. They must match"
        self.weights = weights
    
    def __call__(self, x: torch.tensor) -> torch.tensor:
        outputs = torch.zeros((self.n, x.size(0), self.nclasses))
        
        if self.weights is None:
            for i, model in enumerate(self.models):
                outputs[i] = model(x)
            outputs = outputs.mean(dim=0)
        else:
            for i, (model, w) in enumerate(zip(self.models, self.weights)):
                outputs[i] = w * model(x)
            outputs = outputs.sum(dim=0)
            outputs /= self.weights.sum()
    
    def predict(self, x: torch.tensor) -> torch.tensor:
        outputs = self.__call__(x)
        return outputs.argmax(dim=1)


class Voting:
    def __init__(self, models: list) -> None:
        self.models = models
        self.n_models = len(models)
        self.nclasses = models[0].nclasses
    
    def __call__(self, x: torch.tensor) -> torch.tensor:
        outputs = torch.zeros((x.size(0), self.nclasses))
        for i, model in enumerate(self.models):
            for j, p in enumerate(model(x)):
                outputs[j, p.argmax()] += 1
        return outputs
    
    def predict(self, x):
        outputs = self.__call__(x)
        return outputs.argmax(dim=1)


class Stacking:
    pass


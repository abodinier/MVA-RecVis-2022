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
        outputs = torch.zeros((self.n_models, x.size(0), self.nclasses))
        
        if self.weights is None:
            for i, model in enumerate(self.models):
                outputs[i] = model(x)
            outputs = outputs.mean(dim=0)
        else:
            for i, (model, w) in enumerate(zip(self.models, self.weights)):
                outputs[i] = w * model(x)
            outputs = outputs.sum(dim=0)
            outputs /= self.weights.sum()
        
        return outputs
    
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
    def __init__(self, bag_of_models, stacking_model, **kwargs) -> None:
        self.bag_of_models = bag_of_models
        self.n_models = len(bag_of_models)
        self.nclasses = bag_of_models[0].nclasses
        self.stacing_model = stacking_model
    
    def prepare_x_bag(self, X):
        for i, model in enumerate(self.bag_of_models):
            if i == 0:
                x_bag = model(X).detach().numpy()
            else:
                x_bag = np.hstack([x_bag, model(X).detach().numpy()])
        
        return x_bag
    
    def fit(self, X, y):
        x_bag = self.prepare_x_bag(X)
        
        self.stacing_model.fit(x_bag, y.detach().numpy())
    
    def predict(self, X):
        x_bag = self.prepare_x_bag(X)
        
        return self.stacing_model.predict(x_bag)


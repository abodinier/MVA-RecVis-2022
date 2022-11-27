import tqdm
import torch
import numpy as np
import pandas as pd
from torch import optim
from metrics import Metrics
from torch.optim import SGD, Adam, Adadelta


def get_optim(optim_name, model, **kwargs):
    if optim_name.lower() == "sgd":
        return optim.SGD(
            params=model.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"]
        )
    if optim_name.lower() == "adam":
        return optim.Adam(params=model.parameters(), lr=kwargs["lr"])


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        use_cuda=False,
        logger=None,
        experiment_dir=None,
        args=None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.use_cuda = use_cuda
        self.logger = logger
        self.experiment_dir = experiment_dir
        self.args = args

        if self.use_cuda:
            print("Using GPU")
            self.model.cuda()
        else:
            print("Using CPU")

        self.epoch = 0
        self.best_loss = np.inf

        self.train_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": []}
        self.val_metrics = {"loss": [], "accuracy": [], "precision": [], "recall": []}

    def checkpoint(self, val_loss):
        if self.experiment_dir:
            ckp = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_loss": self.best_loss,
            }

            torch.save(obj=ckp, f=self.experiment_dir / f"models/ckp_{self.epoch}.pt")

            learning_curve_df = pd.DataFrame()
            for k, v in self.train_metrics.items():
                learning_curve_df[f"train_{k}"] = v
            for k, v in self.val_metrics.items():
                learning_curve_df[f"val_{k}"] = v
            learning_curve_df.to_csv(
                self.experiment_dir / "metrics/learning_curves.csv"
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                ckp_best = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                    "best_loss": self.best_loss,
                }
                torch.save(obj=ckp_best, f=self.experiment_dir / f"models/ckp_best.pt")

    def update_metrics(self, train_metrics, val_metrics):
        for k, v in train_metrics.items():
            self.train_metrics[k].append(v)
        for k, v in val_metrics.items():
            self.val_metrics[k].append(v)

    def train(self, epochs):
        while self.epoch < epochs:
            train_metrics = self.training_epoch()
            val_metrics = self.validation_epoch()
            self.update_metrics(train_metrics=train_metrics, val_metrics=val_metrics)
            print(
                f"[Epoch {self.epoch}/{epochs}]\n\
                Train {train_metrics['loss']:.2f} / {100 * train_metrics['accuracy']:.2f}%\n\
                Val {val_metrics['loss']:.2f} / {100 * val_metrics['accuracy']:.2f}%"
            )
            if self.logger:
                for k, v in train_metrics.items():
                    self.logger.add_scalar("train/" + k, v, self.epoch)
                for k, v in val_metrics.items():
                    self.logger.add_scalar("val/" + k, v, self.epoch)
            val_loss = val_metrics.get("loss", np.inf)
            self.checkpoint(val_loss)
            self.scheduler.step()
            self.epoch += 1

    def training_epoch(self):
        self.model.train()
        metric_cls = Metrics()
        for batch_idx, (x, y) in enumerate(self.train_dataloader):

            if self.use_cuda:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()

            if self.args.model_name == "inception":
                output, aux_outputs = self.model(x)
                loss1 = self.criterion(output, y)
                loss2 = self.criterion(aux_outputs, y)
                loss = loss1 + 0.4 * loss2
            else:
                output = self.model(x)
                loss = self.criterion(input=output, target=y)

            loss.backward()

            self.optimizer.step()

            metric_cls.update_on_batch(y_true=y, output=output, loss=loss.item())

        return metric_cls.get_metrics()

    @torch.no_grad()
    def validation_epoch(self):
        self.model.eval()

        metric_cls = Metrics()

        for batch_idx, (x, y) in enumerate(self.val_dataloader):

            if self.use_cuda:
                x, y = x.cuda(), y.cuda()

            output = self.model(x)

            loss = self.criterion(input=output, target=y)

            metric_cls.update_on_batch(y_true=y, output=output, loss=loss.item())

        return metric_cls.get_metrics()

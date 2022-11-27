import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


class Metrics:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.accuracies = []
        self.precision_scores = []
        self.recall_scores = []
        self.losses = []
        self.auc_scores = []

    def update_on_batch(self, y_true, output, loss):
        preds = torch.argmax(output, dim=1).numpy()
        self.accuracies.append(accuracy_score(y_true, y_pred=preds))
        self.precision_scores.append(
            precision_score(y_true, y_pred=preds, average="macro", zero_division=0)
        )
        self.recall_scores.append(
            recall_score(y_true, y_pred=preds, average="macro", zero_division=0)
        )
        self.losses.append(loss)

    def get_metrics(self):
        metrics = {}
        metrics[f"{self.prefix}accuracy"] = np.mean(self.accuracies)
        metrics[f"{self.prefix}precision"] = np.mean(self.precision_scores)
        metrics[f"{self.prefix}recall"] = np.mean(self.recall_scores)
        metrics[f"{self.prefix}loss"] = np.mean(self.losses)
        return metrics

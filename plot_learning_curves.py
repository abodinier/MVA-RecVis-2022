import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

EXPERIMENT_PATH = Path(sys.argv[1])

df = pd.read_csv(EXPERIMENT_PATH/"metrics/learning_curves.csv")

plt.figure(figsize=(10, 10))
df[["train_loss", "val_loss"]].plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig(EXPERIMENT_PATH/"plots/loss.jpg")

plt.figure(figsize=(10, 10))
df[["train_accuracy", "val_accuracy"]].plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.savefig(EXPERIMENT_PATH/"plots/accuracy.jpg")
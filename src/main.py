import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import get_model, CNN, VGG, ResNet, Inception, ViT
from training import Trainer, get_optim
from data import get_data_transform
from metrics import Metrics


# Training settings
parser = argparse.ArgumentParser(description="RecVis A3 training script")
parser.add_argument(
    "--nclasses",
    type=int,
    default=20,
)
parser.add_argument(
    "--data",
    type=str,
    default="bird_dataset",
    metavar="D",
    help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
)
parser.add_argument(
    "--image-size",
    type=int,
    default=224,
    metavar="S",
    help="Image size in pixels",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="resnet",
    metavar="M",
    help="Model <cnn|vgg|resnet|inception|vit>",
)
parser.add_argument(
    "--version",
    type=int,
    metavar="V",
    help="Model version",
)
parser.add_argument(
    "--batch-norm",
    type=int,
    metavar="BN",
    help="Use batch norm",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="B",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--optim-name",
    type=str,
    default="sgd",
    metavar="O",
    help="optimizer",
)
parser.add_argument(
    "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.01)"
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="learning rate (default: 0.5)",
)
parser.add_argument("--lr-step", type=int, default=10, help="Reduce LR each lr_step")
parser.add_argument("--lr-gamma", type=float, default=0.5, help="LR decay")
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--data-augmentation",
    type=int,
    default=0,
    help="data augmentation",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    metavar="d",
    help="Dropout ratio",
)
parser.add_argument(
    "--fc-dim",
    type=int,
    default=50,
    help="Last hidden layer dimension",
)
parser.add_argument(
    "--freeze-weights",
    type=int,
    default=0,
    help="Freeze backbone weights",
)
parser.add_argument(
    "--exp-name",
    type=str,
    default="",
    help="Exp name",
)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
exp_name = args.exp_name if args.exp_name != "" else "experiment"
EXPERIMENT = Path(f"EXPERIMENTS/{exp_name}_{timestamp}")
METRICS = EXPERIMENT / "metrics/"
PLOTS = EXPERIMENT / "plots/"
MODELS = EXPERIMENT / "models/"
EXPERIMENT.mkdir(exist_ok=True)
METRICS.mkdir()
PLOTS.mkdir()
MODELS.mkdir()

logger = SummaryWriter(log_dir=EXPERIMENT / "tensorboard")
with open(MODELS / "parameters.json", "w") as f:
    json.dump(args.__dict__, f)

# Data initialization and loading
data_transforms = get_data_transform(**args.__dict__)

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
    batch_size=args.batch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + "/val_images", transform=data_transforms),
    batch_size=args.batch_size,
    shuffle=False,
)

model = get_model(**args.__dict__)
optimizer = get_optim(model=model, **args.__dict__)
scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
criterion = torch.nn.CrossEntropyLoss(reduction="mean")

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    use_cuda=use_cuda,
    logger=logger,
    experiment_dir=EXPERIMENT,
    args=args,
)

trainer.train(epochs=args.epochs)

metrics = {}
train_metrics = Metrics()
for batch_idx, (x, y) in enumerate(train_loader):
    output = model(x)
    loss = criterion(output, y)
    train_metrics.update_on_batch(y_true=y, output=output, loss=loss.item())
metrics["train"] = train_metrics.get_metrics()

actuals, predicteds = [], []
test_metrics = Metrics()
for batch_idx, (x, y) in enumerate(val_loader):
    actuals += y.detach().numpy().tolist()
    output = model(x)
    loss = criterion(output, y)
    preds = torch.argmax(output, 1).detach().numpy().tolist()
    predicteds += preds
    test_metrics.update_on_batch(y_true=y, output=output, loss=loss.item())
metrics["val"] = test_metrics.get_metrics()

with open(METRICS / "metrics.json", "w") as f:
    json.dump(metrics, f)
# DVC
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

with open(PLOTS / "confusion.csv", "w") as f:
    f.write("actual,predicted\n")
    for actual, preds in zip(actuals, predicteds):
        f.write(f"{actual},{preds}\n")
# DVC
with open("confusion.csv", "w") as f:
    f.write("actual,predicted\n")
    for actual, preds in zip(actuals, predicteds):
        f.write(f"{actual},{preds}\n")

ConfusionMatrixDisplay.from_predictions(
    y_true=actuals,
    y_pred=predicteds,
    normalize="true",
)

plt.savefig(PLOTS / "confusion_matrix.jpg")

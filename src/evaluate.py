import os
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import PIL.Image as Image

from model import load_from_exp
from data import get_data_transform


parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
parser.add_argument(
    "--data",
    type=str,
    default="data",
    metavar="D",
    help="folder where data is located. test_images/ need to be found in the folder",
)
parser.add_argument(
    "--image-size",
    type=int,
    default=299,
)
parser.add_argument(
    "--data-augmentation",
    type=int,
    default=0,
)
parser.add_argument(
    "--model_dir",
    type=str,
    metavar="M",
    default="models/ckp",
    help="the model file to be evaluated. Usually it is of the form model_X.pth",
)
parser.add_argument(
    "--outfile",
    type=str,
    default="kaggle.csv",
)

args = parser.parse_args()
test_dir = args.data + "/test_images/mistery_category"
data_transforms = get_data_transform(**args.__dict__)
use_cuda = torch.cuda.is_available()

model = load_from_exp(Path(args.model_dir))


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if "jpg" in f:
        data = data_transforms(pil_loader(test_dir + "/" + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print(
    "Succesfully wrote "
    + args.outfile
    + ", you can upload this file to the kaggle competition website"
)

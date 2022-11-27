import os
import json
import torch
import warnings
from tqdm import tqdm
import PIL.Image as Image
from pathlib import Path

from model import CNN, ResNet, Inception, VGG, ViT, load_from_exp
from data import get_data_transform
from ensembling import Averaging, Voting, Stacking

nclasses = 20

resnet = load_from_exp(Path("EXPERIMENTS/resnet_2022-11-27-00-56"))
vgg = load_from_exp(Path("EXPERIMENTS/vgg_2022-11-27-00-55"))
inception = load_from_exp(Path("EXPERIMENTS/inception_2022-11-27-00-54"))
bag_of_models = Averaging(models=[resnet, vgg, inception], weights=torch.tensor([1, 1, 1]))


test_dir = "../data/test_images/mistery_category"
data_transforms = get_data_transform(image_size=384, data_augmentation=0)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")

warnings.simplefilter('ignore')
output_file = open("kaggle_ensemble.csv", "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if "jpg" in f:
        data = data_transforms(pil_loader(test_dir + "/" + f))
        data = data.view(1, data.size(0).size(1), data.size(2))
        output = bag_of_models(data)
        pred = output.argmax().item()
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

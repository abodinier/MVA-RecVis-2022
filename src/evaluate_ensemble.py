import os
import json
import torch
from tqdm import tqdm
import PIL.Image as Image

from model import CNN, ResNet, Inception, VGG, ViT
from data import get_data_transform

nclasses = 20

resnet_params = json.load(
    open("experiment_2022-11-26-23-55/models/parameters.json", "r")
)
resnet = ResNet(**resnet_params)
resnet.load_state_dict(
    torch.load("experiment_2022-11-26-23-55/models/ckp_0.pt")["model"]
)

vgg_params = json.load(open("experiment_2022-11-26-23-55/models/parameters.json", "r"))
vgg = ResNet(**vgg_params)
vgg.load_state_dict(torch.load("experiment_2022-11-26-23-55/models/ckp_0.pt")["model"])

inception_params = json.load(
    open("experiment_2022-11-26-23-55/models/parameters.json", "r")
)
inception = ResNet(**inception_params)
inception.load_state_dict(
    torch.load("experiment_2022-11-26-23-55/models/ckp_0.pt")["model"]
)

bag_of_models = [resnet, vgg, inception]

use_cuda = torch.cuda.is_available()


test_dir = "../data/test_images/mistery_category"
data_transforms = get_data_transform(image_size=220, data_augmentation=0)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


output_file = open("kaggle_ensemble.csv", "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if "jpg" in f:
        data = data_transforms(pil_loader(test_dir + "/" + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        outputs = torch.zeros((len(bag_of_models), nclasses))
        for i, model in enumerate(bag_of_models):
            output = model(data)[0]
            outputs[i] = output
        output = outputs.mean(dim=0)
        pred = output.argmax().item()
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

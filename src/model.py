import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTFeatureExtractor, ViTForImageClassification


class Activations:
    RELU = nn.ReLU()
    SIGMOID = nn.Sigmoid()
    TANH = nn.Tanh()


class Pooling:
    MAX = nn.MaxPool2d(2)
    AVG = nn.AvgPool2d(2)


def freeze_weights_model(model):
    for param in model.parameters():
        param.requires_grad = False


def get_model(**kwargs):
    if kwargs["model_name"].lower() == "cnn":
        return CNN(**kwargs)
    if kwargs["model_name"].lower() == "resnet":
        return ResNet(**kwargs)
    if kwargs["model_name"].lower() == "vgg":
        return VGG(**kwargs)
    if kwargs["model_name"].lower() == "inception":
        return Inception(**kwargs)
    if kwargs["model_name"].lower() == "vit":
        return ViT(**kwargs)


class CNN(nn.Module):
    def __init__(
        self,
        nclasses=20,
        activation=Activations.RELU,
        pooling=Pooling.MAX,
        dropout_ratio=0,
        fc_dim=50,
        **kwargs,
    ):
        super(CNN, self).__init__()
        self.nclasses = nclasses
        self.dropout_ratio = dropout_ratio
        self.fc_dim = fc_dim

        self.activation = activation
        self.pooling = pooling

        self.dropout = nn.Dropout(dropout_ratio)

        self.layers = nn.Sequential(
            *[
                nn.Conv2d(3, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Conv2d(20, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Flatten(),
                nn.LazyLinear(fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(fc_dim, nclasses),
            ]
        )

    def forward(self, x):
        return self.layers(x)


class ResNet(nn.Module):
    def __init__(self, nclasses=20, freeze_weights=0, version=152, **kwargs):
        super(ResNet, self).__init__()
        self.nclasses = nclasses
        assert version in (
            18,
            34,
            50,
            101,
            152,
        ), f"There is no resnet{version}"

        resnet = eval(f"models.resnet{version}(pretrained=True)")
        if freeze_weights:
            freeze_weights_model(resnet)
        resnet.fc = nn.LazyLinear(nclasses)

        self.layers = nn.Sequential(
            *[
                resnet,
            ]
        )

    def forward(self, x):
        return self.layers(x)


class Inception(nn.Module):
    def __init__(self, nclasses=20, freeze_weights=0, **kwargs):
        super(Inception, self).__init__()
        self.nclasses = nclasses

        inception = models.inception_v3(pretrained=True)
        if freeze_weights:
            freeze_weights_model(inception)
        inception.AuxLogits.fc = nn.LazyLinear(nclasses)
        inception.fc = nn.LazyLinear(nclasses)

        self.layers = nn.Sequential(
            *[
                inception,
            ]
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        outputs = self.forward(x)[0]
        preds = torch.argmax(outputs, dim=0)
        return preds


class VGG(nn.Module):
    def __init__(
        self, nclasses=20, freeze_weights=0, version="", batch_norm=False, **kwargs
    ):
        super(VGG, self).__init__()
        self.nclasses = nclasses
        assert version in (11, 13, 16, 19), f"There is no VGG{version}"

        vgg = eval(
            f"models.vgg{str(version) if batch_norm == False else str(version) + '_bn'}(pretrained=True)"
        )
        if freeze_weights:
            freeze_weights_model(vgg)
        vgg.classifier[6] = nn.LazyLinear(nclasses)

        self.layers = nn.Sequential(
            *[
                vgg,
            ]
        )

    def forward(self, x):
        return self.layers(x)


class ViT(nn.Module):
    def __init__(
        self,
        nclasses=20,
        model_name_or_path="google/vit-base-patch16-224-in21k",
        **kwargs,
    ):
        super(ViT, self).__init__()
        self.nclasses = nclasses

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.fc = nn.LazyLinear(nclasses)

    def forward_from_pil(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        return logits

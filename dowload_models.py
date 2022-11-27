import os
import ssl
import numpy as np
from torchvision import models

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

model = models.resnet152(pretrained=True)
model = models.resnet18(pretrained=True)
model = models.resnet34(pretrained=True)
model = models.vgg16_bn(pretrained=True)
model = models.vgg16_bn(pretrained=True)
model = models.inception_v3(pretrained=True)

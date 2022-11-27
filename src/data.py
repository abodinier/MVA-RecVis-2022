import torchvision.transforms as transforms


def get_data_transform(**kwargs):
    size = kwargs["image_size"]
    augment = kwargs["data_augmentation"]

    if augment:
        return transforms.Compose(
            [
                transforms.Resize((size + 50, size + 50)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomAffine(degrees=90),
                transforms.RandomCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=0.1),
                transforms.RandomErasing(ratio=[0.1, 0.1]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

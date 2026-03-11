from torchvision import transforms

MEAN_CLIP = (0.48145466, 0.4578275, 0.40821073)
STD_CLIP = (0.26862954, 0.26130258, 0.27577711)

# --------------------------------------------------
# Transform
# --------------------------------------------------
def build_transform(image_size, mean=MEAN_CLIP, std=STD_CLIP):

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

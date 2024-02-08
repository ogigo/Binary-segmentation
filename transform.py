import albumentations as A

def get_transforms():
    return A.Compose(
        [
            A.HueSaturationValue( # Change colors
                p=1.0,
                hue_shift_limit=(-20, 20),
                sat_shift_limit=(-30, 30),
                val_shift_limit=(-20, 20),
            ),
            A.HorizontalFlip(p=0.5),
        ],
        p=1.0
    )
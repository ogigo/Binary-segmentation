import torch
from torch.utils.data import Dataset,DataLoader
from transform import get_transforms
from segmentation_models_pytorch.encoders import get_preprocessing_fn


images="collection of images"
masks="collection of masks"

N_IMAGES = 512
TRAIN_IMAGE_SIZE = 512
INPUT_IMAGE_SIZE = (1920, 1080)

indexes = list(range(N_IMAGES))
train_indexes = indexes[: int(N_IMAGES * 0.8)]
valid_indexes = indexes[int(N_IMAGES * 0.8) :]
preprocess_input = get_preprocessing_fn("resnet34", pretrained="imagenet")

class CustomDataset(Dataset):
    def __init__(self, indexes, transform=None, preprocess=None):
        self.indexes = indexes
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        _index = self.indexes[index]

        image = images[_index]
        mask = masks[_index]

        if self.transform:
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]

        if self.preprocess:
            image = self.preprocess(image)

        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)

        image = image.permute(2, 0, 1)
        mask = mask.unsqueeze(0)

        return {"image": image, "mask": mask}
    
    
train_dataset = CustomDataset(train_indexes, transform=get_transforms(), preprocess=preprocess_input)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

valid_dataset = CustomDataset(valid_indexes, preprocess=preprocess_input)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
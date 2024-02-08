import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

state_dict=torch.load("last_model.pt",map_location=torch.device('cpu'))
model.load_state_dict(state_dict["model_state_dict"])
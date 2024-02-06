import base64
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import io
from model import model
import binascii


def segment_image(image_path,image_size):
    cur_image = cv2.imread(image_path)
    cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)
    cur_image = cv2.resize(cur_image, (image_size, image_size))

    image = torch.tensor(cur_image, dtype=torch.float)
    image = image.permute(2, 0, 1)
    image=image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    segmented_image = output.squeeze()
    segmented_image=segmented_image.detach().cpu().numpy()
    segmented_image = 1 - segmented_image

    # Save the segmented image
    segmented_image_path = f"{image_path.replace('.', '_segmented2.')}"
    Image.fromarray(segmented_image.astype(np.uint8)).save(segmented_image_path)

    return segmented_image_path
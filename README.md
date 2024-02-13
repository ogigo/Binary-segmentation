# Binary Image Segmentation Project

## You can see the end product on the link bellow-
[Watch the video](https://youtu.be/i0ffV9FnE1s)


## Introduction

Welcome to the Football Player Segmentation Project! This project focuses on segmenting football players from images using the UNet model implemented in the `segment_models_pytorch` library. The UNet architecture is well-suited for image segmentation tasks, offering high accuracy and efficiency.

## Objective

The main objective of this project is to train a UNet model to accurately segment football players from images captured during matches. By leveraging the UNet architecture and the power of PyTorch, we aim to develop a robust segmentation model capable of handling varying lighting conditions, player poses, and background complexities.

## Key Features

- **UNet Model**: We utilize the UNet architecture, known for its effectiveness in semantic segmentation tasks, to segment football players from images.

- **PyTorch Implementation**: The model is implemented using PyTorch, a popular deep learning framework known for its flexibility and ease of use.

- **Transfer Learning**: We explore transfer learning techniques to leverage pre-trained UNet models for improved segmentation performance and faster convergence.

- **Data Augmentation**: We employ data augmentation techniques such as random flips, rotations, and scaling to enhance model generalization and robustness.

## Getting Started

### Installation

To install the required dependencies, run:

```bash
git clone https://github.com/ogigo/Binary-segmentation.git
cd Binary-segmentation
pip install -r requirements.txt

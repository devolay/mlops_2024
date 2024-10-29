import os
import torch

from torchvision import models

model = models.resnet18(pretrained=True)
torch.save(model, f"{os.environ['MODEL_PATH']}/model.pth")
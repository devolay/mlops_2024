import os
import torch

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
torch.save(model, os.environ['MODEL_PATH'])
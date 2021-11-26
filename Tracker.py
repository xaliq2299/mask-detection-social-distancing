import cv2

from torchvision import models, transforms

import torch
import torch.nn as nn

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

path_to_video = "./videos/Macron_tracking.mp4"


#################################################################################
# LOAD THE MODELS
#################################################################################

MaskRecognition_model = models.resnet50(pretrained=True)
for param in MaskRecognition_model.parameters():
    param.requires_grad = False
MaskRecognition_model.fc = nn.Sequential(
    nn.Linear(2048,128),
    nn.ReLU(),
    nn.Linear(128,1),
    nn.Sigmoid())
MaskRecognition_model.to(device)
MaskRecognition_model = torch.load("./model/MaskRecognitionRSN50.pt")
MaskRecognition_model.eval()

def hasMask(face_img, model):
    face_img_tensor = transforms.Resize((64,64))(torch.Tensor(face_img).permute(2,0,1))
    face_img_tensor = face_img_tensor.reshape((1,3,64,64)).to(device)
    return torch.round(model(face_img_tensor)).item() == 1

#################################################################################
# LAUNCH THE TRACKING
#################################################################################

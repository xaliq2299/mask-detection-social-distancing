import cv2

from torchvision import models, transforms

import torch
import torch.nn as nn



# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

# if string: path to video
# if 0 (int): use webcam
video_path = "./data/Videos/macron.mp4"
# video_path = 0



#################################################################################
# LOAD THE MODELS
#################################################################################

FaceDetection_model_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# FaceDetection_model_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def detectFaces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return FaceDetection_model_frontal.detectMultiScale(gray_img, 1.1, 4)
    # return FaceDetection_model_profile.detectMultiScale(gray_img, 1.1, 4)

MaskRecognition_model = models.resnet50(pretrained=True)
for param in MaskRecognition_model.parameters():
    param.requires_grad = False
MaskRecognition_model.fc = nn.Sequential(
    nn.Linear(2048,1024),
    nn.ReLU(),
    nn.Linear(1024,256),
    nn.ReLU(),
    nn.Linear(256,1),
    nn.Sigmoid())
MaskRecognition_model.to(device)
MaskRecognition_model.load_state_dict(torch.load("./models/MaskRecognitionRSN50.pt"))
MaskRecognition_model.eval()

def hasMask(face_img):
    face_img_tensor = transforms.Resize((64,64))(torch.Tensor(face_img).permute(2,0,1))
    face_img_tensor = face_img_tensor.reshape((1,3,64,64)).to(device)
    return torch.round(MaskRecognition_model(face_img_tensor)).item() == 1



#################################################################################
# LIVE TRACKING
#################################################################################

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

_, frame = cap.read()
cv2.imshow("Tracker", frame)

while cv2.getWindowProperty("Tracker", 0) >= 0:

    _, frame = cap.read()


    cv2.imshow("Tracker", frame)
    
    c = cv2.waitKey(100)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

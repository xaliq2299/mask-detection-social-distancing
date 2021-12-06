import cv2

from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import torch.nn as nn
import torchvision



# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

# if string: path to video
# if 0 (int): use webcam
video_path = "./data/Videos_Raw/macron.mp4"
# video_path = 0



#######################################################################################
# LOAD THE MODELS
#######################################################################################

###############################
### HAAR CASCADE + RESNET50 ###
###############################

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


##############################
### FASTER-RCNN (RESNET50) ###
##############################

def get_rcnn_model(nb_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nb_classes)
    return model

modelRCNN = get_rcnn_model(nb_classes=4)
modelRCNN.to(device)
modelRCNN.load_state_dict(torch.load("./models/MaskRecognitionFasterRCNN.pt"))
modelRCNN.eval()

def getboxesRCNN(frame):

    height, width, nb_channel = frame.shape
    model_input = transforms.Resize((256,256))(torch.Tensor(frame).permute(2,0,1))
    model_input = model_input.reshape((1,3,256,256)).to(device)
    target = modelRCNN(model_input)[0]

    size = len(target["boxes"])
    boxes = []
    
    for i in range(size):
        box = target["boxes"][i]
        label = int(target["labels"][i])
        xmin = int(width*box[0]/256)
        ymin = int(height*box[1]/256)
        xmax = int(width*box[2]/256)
        ymax = int(height*box[3]/256)
        boxes.append((xmin,ymin,xmax,ymax,label))
    
    return boxes

#######################################################################################
# LIVE TRACKING
#######################################################################################



cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

_, frame = cap.read()
overlay = frame.copy()
output = frame.copy()
cv2.imshow("Tracker", frame)

while cv2.getWindowProperty("Tracker", 0) >= 0:

    boxes = getboxesRCNN(frame)

    for (xmin,ymin,xmax,ymax,label) in boxes:
        if label == 1:
            cv2.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
        elif label == 2:
            cv2.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,127,127), 2)
        else:
            cv2.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
    
    output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    cv2.imshow("Tracker", output)
    
    c = cv2.waitKey(1)
    if c == 27:
        break

    _, frame = cap.read()
    overlay = frame.copy()
    output = frame.copy()

cap.release()
cv2.destroyAllWindows()

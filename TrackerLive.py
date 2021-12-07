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
# video_path = "./data/Videos_Raw/macron.mp4"
video_path = 0



#######################################################################################
# LOAD THE MODELS
#######################################################################################

###############################
### HAAR CASCADE + RESNET18 ###
###############################

FaceDetection_model_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FaceDetection_model_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

modelRSN18 = models.resnet18(pretrained=True)
for param in modelRSN18.parameters():
    param.requires_grad = False
modelRSN18.fc = nn.Sequential(
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Linear(256,64),
    nn.ReLU(),
    nn.Linear(64,3),
    nn.Softmax())
modelRSN18.to(device)
modelRSN18.load_state_dict(torch.load("./models/MaskRecognitionRSN18.pt", map_location=torch.device('cpu')))
modelRSN18.eval()

def get_boxes_RSN18(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes_frontal = FaceDetection_model_frontal.detectMultiScale(gray_img, 1.1, 4)
    boxes_profile = FaceDetection_model_profile.detectMultiScale(gray_img, 1.1, 4)
    results = []
    for (x,y,w,h) in boxes_frontal:
        model_input = transforms.Resize((256,256))(torch.Tensor(img[y:y+h,x:x+w]).permute(2,0,1))
        model_input = model_input.reshape((1,3,256,256)).to(device)
        label = torch.argmax(modelRSN18(model_input)).item()+1
        results.append((x,y,x+w,y+h,label))
    for (x,y,w,h) in boxes_profile:
        model_input = transforms.Resize((256,256))(torch.Tensor(img[y:y+h,x:x+w]).permute(2,0,1))
        model_input = model_input.reshape((1,3,256,256)).to(device)
        label = torch.argmax(modelRSN18(model_input)).item()+1
        results.append((x,y,x+w,y+h,label))
    return results


##############################
### FASTER-RCNN (RESNET50) ###
##############################

modelRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = modelRCNN.roi_heads.box_predictor.cls_score.in_features
modelRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
modelRCNN.to(device)
modelRCNN.load_state_dict(torch.load("./models/MaskRecognitionFasterRCNN.pt", map_location=torch.device('cpu')))
modelRCNN.eval()

def get_boxes_RCNN(frame):

    height, width, nb_channel = frame.shape
    model_input = transforms.Resize((256,256))(torch.Tensor(frame).permute(2,0,1))
    model_input = model_input.reshape((1,3,256,256)).to(device)
    target = modelRCNN(model_input)[0]
    boxes = []
    
    for i in range(len(target["boxes"])):
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

    boxes = get_boxes_RSN18(frame)
    # boxes = get_boxes_RCNN(frame)

    for (xmin,ymin,xmax,ymax,label) in boxes:
        if label == 1:
            cv2.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
        elif label == 2:
            cv2.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,256,256), 2)
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

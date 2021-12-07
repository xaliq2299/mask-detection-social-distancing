import cv2 as cv
import numpy as np
import os

from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn


from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import SocialDistancing
import ipywidgets as widgets


def hasMask(face_img, model):
    '''
    Takes as input face image and model
    Returns if the face is present with/without mask
    '''
    face_img_tensor = transforms.Resize((64,64))(torch.Tensor(face_img).permute(2,0,1))
    face_img_tensor = face_img_tensor.reshape((1,3,64,64)).to(device)
    return torch.round(model(face_img_tensor)).item() == 1

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


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

#################################################################################
# PARAMETERS
#################################################################################
# todo: change after maybe. User args
INPUT_PATH = "data/Videos_Raw/Macron.mp4"
WEIGHTS_PATH = 'multi-object-tracker/Yolov3/yolov3.weights'
CONFIG_FILE_PATH = 'multi-object-tracker/Yolov3/yolov3.cfg'
LABELS_PATH = "multi-object-tracker/examples/pretrained_models/yolo_weights/coco_names.json"
OUTPUT_PATH = 'data/Videos_Processed/Macron-RCNN.mp4'
# todo: for output, create directory called results and save the output there

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2
DRAW_BOUNDING_BOXES = True
USE_GPU = False
MIN_DISTANCE = 50 # needed for social distancing 2.3 the best so far
# colors
RED   = (0, 0, 255)
BLUE  = (255, 0, 0)

#################################################################################
# LOAD THE MODELS
#################################################################################

# 1. Mask Recognition model (Faster R-CNN)
modelRCNN = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = modelRCNN.roi_heads.box_predictor.cls_score.in_features
modelRCNN.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
modelRCNN.to(device)
modelRCNN.load_state_dict(torch.load("./models/MaskRecognitionFasterRCNN.pt", map_location=torch.device('cpu')))
modelRCNN.eval()

# 2. Human Detection model (Yolov3)
HumanDetection_model = YOLOv3(
    weights_path=WEIGHTS_PATH,
    configfile_path=CONFIG_FILE_PATH,
    labels_path=LABELS_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    nms_threshold=NMS_THRESHOLD,
    draw_bboxes=DRAW_BOUNDING_BOXES,
    use_gpu=USE_GPU
)

# TODO: face detection via what?
# 3. Face Detection model (from OpenCV)
FaceDetection_model = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
#hog = cv.HOGDescriptor()
#hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# 4. SORT TRACKER
tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)


#################################################################################
# LAUNCH THE TRACKING
#################################################################################

cap = cv.VideoCapture(INPUT_PATH)
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

# fourcc = cv.VideoWriter_fourcc(*'MP4V') # TIVX/DIVX for avi format
# fourcc = cv.VideoWriter_fourcc(*'MPEG')
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# fourcc = cv.VideoWriter_fourcc('M','J','P','G') # ('M','P,'E','G')
fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = 10
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))


# while True:
num_frames = 50
for i in range(num_frames):
    print('frame', i+1)
    ok, frame = cap.read()

    if not ok:
        print("Cannot read the video feed.")
        break

    overlay = frame.copy()
    output = frame.copy() # todo
    frame = cv.resize(frame, (frame_width, frame_height))
    # print(overlay.shape)

    #################################################################################
    # RUNNING MODEL FOR MASK RECOGNITION (face+mask recognition models)
    #################################################################################
    face_boxes = get_boxes_RCNN(frame)
    # print('face boxes:', face_boxes)
    
    for (xmin,ymin,xmax,ymax,label) in face_boxes:
        if label == 1:
            cv.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
        elif label == 2:
            cv.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,127,127), 2)
        else:
            cv.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,255,0), 2)





    # for i, (x,y,w,h) in enumerate(face_boxes):
    #     if hasMask(frame[y:y+h,x:x+h], MaskRecognition_model):
    #         print("Mask!")
    #         cv.rectangle(overlay, (x,y), (x+w,y+h), BLUE, 3)
    #     else:
    #         print("No mask")
    #         cv.rectangle(overlay, (x,y), (x+w,y+h), RED, 3)



    # output = cv.addWeighted(overlay, 0.05, output, 0.90, 0, output) # todo: needed?

    #################################################################################
    # SORT TRACKING (human detection model)
    #################################################################################
    human_boxes, confidences, class_ids = HumanDetection_model.detect(frame)
    # human_boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    # print('class ids:', class_ids)
    
    tracks = tracker.update(human_boxes, confidences, class_ids)
    # overlay = HumanDetection_model.draw_bboxes(overlay, human_boxes, confidences, class_ids)
    overlay = draw_tracks(overlay, tracks)

    #################################################################################
    # SOCIAL DISTANCING (human detection model + simple heuristics)
    #################################################################################
    # directory = 'disparity_maps/'
    # if not os.path.exists(directory):
    #    os.makedirs(directory)
    # cv.imwrite(directory + 'img.jpg', overlay) # maybe smth else

    # output_name = INPUT_PATH.split('/')[1].split('.')[0] + '_frame_' + str(i) + '.jpeg'
    # print(output_name)
    # disparity.disparity_map_calc(overlay, output_name)
    # os.system('python3 monodepth2/test_simple.py --image_path disparity_maps/img.jpg --model_name mono+stereo_640x192')

    # img = cv.imread('disparity_maps/img_disp.jpeg')
    # w, h = img.shape[0], img.shape[1]
    # depth = np.zeros([w, h])
    # b = img[:,:,0]
    # g = img[:,:,1]
    # r = img[:,:,2]
    # depth = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # f = 0.8*w # guess for focal length
    # Q = np.float32([[1, 0, 0, -0.5*w],
    #                 [0, -1, 0, 0.5*h],
    #                 [0, 0, 0, -f],
    #                 [0, 0, 1, 0]])
    # output3D = cv.reprojectImageTo3D(depth, Q)
    # todo remove folder at the end




    social_distancing = SocialDistancing.SocialDistancing(MIN_DISTANCE)
    overlay = social_distancing.euclidean(overlay, human_boxes)
    # overlay = social_distancing.depth(overlay, human_boxes, output3D)

    # cv.imshow('', frame) # cv_imshow in Colab
    out.write(overlay) # output

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# releasing everything at the end
cap.release()
out.release()
cv.destroyAllWindows()

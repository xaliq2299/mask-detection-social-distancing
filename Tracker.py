import cv2 as cv
import numpy as np
import os, sys, getopt

from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn

from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import SocialDistancing
import ipywidgets as widgets


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

def get_boxes_RCNN(model, frame):

    height, width, nb_channel = frame.shape
    model_input = transforms.Resize((256,256))(torch.Tensor(frame).permute(2,0,1))
    model_input = model_input.reshape((1,3,256,256)).to(device)
    target = model(model_input)[0]
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



def main(argv):
    image_file_formats = ['.mp4', '.mov', '.wmv', 'avi', '.avchd', '.flv', '.f4v', '.swf', '.mkv', '.webm', '.html5', '.mpeg-2']
    video_file_formats = ['.tiff', '.tif', '.bmp', '.jpg', '.jpeg', ',png', '.gif', '.eps']

    #################################################################################
    # PARAMETERS
    #################################################################################

    INPUT_PATH = ''
    OUTPUT_PATH = ''
    WEIGHTS_PATH = 'models/Yolov3/yolov3.weights'
    CONFIG_FILE_PATH = 'models/Yolov3/yolov3.cfg'
    LABELS_PATH = "multi-object-tracker/examples/pretrained_models/yolo_weights/coco_names.json"
    num_frames = 50 # means ~5 secs

    # social distancing
    PRIOR_BASED_APPROACH = 1
    DEPTH_MAP_ESTIMATOR_APPROACH = 2
    distancing_approach = PRIOR_BASED_APPROACH
    MIN_DISTANCE = 50 # needed for social distancing

    # Yolov3 and SORT
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.2
    DRAW_BOUNDING_BOXES = True
    USE_GPU = False


    #################################################################################
    # PARAMETER PARSING
    #################################################################################
    
    try:
        opts, args = getopt.getopt(argv,"hi:o:f:d:w:c:",["help", "input=", "output", "frames", "distancing", "weights", "config"])
    except getopt.GetoptError:
        print("Usage: Tracker.py -i <inputfile> [-o <outputfile> -f <number of frames> -d <social distancing approach (1 for simple approach,\
                                        2 for depth map estimator> -w <Yolov3 weights path> -c <Yolov3 config file path>]")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: Tracker.py -i <inputfile> [-o <outputfile> -f <number of frames> -d <social distancing approach (1 for simple approach,\
                                        2 for depth map estimator> -w <Yolov3 weights path> -c <Yolov3 config file path>]")
            sys.exit()
        elif opt in ("-i", "--input"):
            INPUT_PATH = arg
            if not os.path.isfile(INPUT_PATH):
                print('[!] Invalid input file path.')
                sys.exit()
            image_format = False
            for i in image_file_formats:
                if i in INPUT_PATH:
                    image_format = True
                    break
            video_format = False
            for i in image_file_formats:
                if i in INPUT_PATH:
                    video_format = True
                    break
            # todo: distinguish between formats
            if video_format == False and image_format == False:
                print('[!] Input file format is not correct.')
                sys.exit()
        elif opt in ("-o", "--output"):
            OUTPUT_PATH = arg
        elif opt in ("-f", "--frames"):
            num_frames = int(arg)
        elif opt in ("-d", "--distancing"):
            distancing_approach = int(arg)
        elif opt in ("-w", "--weights"):
            WEIGHTS_PATH = arg
            if not os.path.isfile(WEIGHTS_PATH):
                print('[!] Invalid YOLOv3 weights file path.')
                sys.exit()
        elif opt in ("-c", "--config"):
            CONFIG_FILE_PATH = arg
            if not os.path.isfile(CONFIG_FILE_PATH):
                print('[!] Invalid YOLOv3 config file path.')
                sys.exit()
        else:
            print("[!] Entered unknown option")
            print("Usage: Tracker.py -i <inputfile> [-o <outputfile> -f <number of frames> -d <social distancing approach (1 for simple approach,\
                                        2 for depth map estimator> -w <Yolov3 weights path> -c <Yolov3 config file path>]")
            sys.exit()

    # checking importance of input file path
    if INPUT_PATH == '':
        print('[!] Input file path required')
        print("Usage: Tracker.py -i <inputfile> [-o <outputfile> -f <number of frames> -d <social distancing approach (1 for simple approach,\
                                        2 for depth map estimator> -w <Yolov3 weights path> -c <Yolov3 config file path>]")
        sys.exit()

    # number of frames
    all_frames_used = False
    if num_frames == -1: # if user has entered -1, then all frames will be processed
        all_frames_used = True
        cap = cv.VideoCapture(INPUT_PATH)
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        print("All frames (i.e.,", num_frames, ") to be processed...")
    else:
        print(num_frames, " frames to be processed...")

    # processing output file path
    if OUTPUT_PATH == '':
        directory = 'data/Videos_Processed/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        nf = num_frames if not all_frames_used else 'all'  # number of frames information to put in the output filename
        OUTPUT_PATH = directory + INPUT_PATH.split('/')[-1].split('.')[0] + '_processed_' + str(nf) + '-frames.mp4'

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
 
    # 3. SORT TRACKER
    tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    

    #################################################################################
    # LAUNCH THE PROCESS
    #################################################################################

    cap = cv.VideoCapture(INPUT_PATH)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 10
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    for i in range(num_frames):
        print('frame', i+1)
        ok, frame = cap.read()

        if not ok:
            print("Cannot read the video feed.")
            break

        overlay = frame.copy()
        output = frame.copy() # todo
        frame = cv.resize(frame, (frame_width, frame_height))


        #################################################################################
        # MASK RECOGNITION
        #################################################################################
        boxes = get_boxes_RCNN(modelRCNN, frame)
        # print('boxes:', boxes)
        
        for (xmin,ymin,xmax,ymax,label) in boxes:
            if label == 1:
                cv.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
            elif label == 2:
                cv.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,127,127), 2)
            else:
                cv.rectangle(overlay, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

        # output = cv.addWeighted(overlay, 0.05, output, 0.90, 0, output) # todo: needed?


        #################################################################################
        # HUMAN DETECTION + HUMAN TRACKING
        #################################################################################
        human_boxes, confidences, class_ids = HumanDetection_model.detect(frame)
        tracks = tracker.update(human_boxes, confidences, class_ids)
        # overlay = HumanDetection_model.draw_bboxes(overlay, human_boxes, confidences, class_ids)
        overlay = draw_tracks(overlay, tracks)


        #################################################################################
        # SOCIAL DISTANCING
        #################################################################################
        if distancing_approach == PRIOR_BASED_APPROACH:
            social_distancing = SocialDistancing.SocialDistancing(MIN_DISTANCE)
            overlay = social_distancing.euclidean(overlay, human_boxes)
        elif distancing_approach == DEPTH_MAP_ESTIMATOR_APPROACH:
            MIN_DISTANCE = 2.3
            social_distancing = SocialDistancing.SocialDistancing(MIN_DISTANCE)

            directory = 'tmp_disparity_maps/'
            if not os.path.exists(directory):
               os.makedirs(directory)
            cv.imwrite(directory + 'img.jpg', overlay)

            output_name = INPUT_PATH.split('/')[1].split('.')[0] + '_frame_' + str(i) + '.jpeg'
            os.system('python3 monodepth2/test_simple.py --image_path tmp_disparity_maps/img.jpg --model_name mono+stereo_640x192')

            img = cv.imread('tmp_disparity_maps/img_disp.jpeg')
            w, h = img.shape[0], img.shape[1]
            
            # depth = np.zeros([w, h])
            disparity = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            f = 0.8*w # guess for focal length
            Q = np.float32([[1, 0, 0, -0.5*w],
                            [0, -1, 0, 0.5*h],
                            [0, 0, 0, -f],
                            [0, 0, 1, 0]])
            depth = cv.reprojectImageTo3D(disparity, Q) # 3D point cloud
            overlay = social_distancing.depth(overlay, human_boxes, depth)
            # os.system('rm -r tmp_disparity_maps/') # todo

        #################################################################################
        out.write(overlay) # todo: output
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # releasing everything at the end
    cap.release()
    out.release()
    cv.destroyAllWindows()
    print("Result saved in the path:", OUTPUT_PATH)


if __name__ == "__main__":
   main(sys.argv[1:])
   exit()

# [**INF634 - COMPUTER VISION PROJECT**](https://moodle.polytechnique.fr/course/view.php?id=13008)
## **Mask Detection and Social Distancing**

*Wearing masks and keeping necessary social distance is essential in the era of COVID.
The goal of this project is to detect if people are wearing masks (in images and over time in
videos) and then compute the distance between them over time. For this, the team will need to
use face detection and tracking techniques, and explore the social distancing effect.*
___

### Setup
After cloning the repository, please follow the commands below in order:
```
cd mask-detection-social-distancing
pip install -r requirements.txt
```
Cloning the Multi-object Tracker repository:
```
git clone https://github.com/xaliq2299/multi-object-tracker.git
cd multi-object-tracker
pip install -r requirements.txt
pip install -e .
pip install ipyfilechooser
```
Downloading a pre-trained Yolov3 model:
```
!./examples/pretrained_models/yolo_weights/get_yolo.sh
mkdir Yolov3/
mv yolov3.weights yolov3.cfg coco.names Yolov3/
mv Yolov3/ ..
cd ..
mkdir models/
mv Yolov3/ models/
```













___

### **Datasets Used**
- [Kaggle: Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection)

___

### **Task 1 | Mask Recognition: Classify people wearing masks**

*Identify if a human face has or has not a mask on (and eventually if it has an incorrectly worn mask.*

First we needed to identify all the faces on a given image. To do so, we used [YOLOv5](https://github.com/ultralytics/yolov5) trained on the Face Mask Detection dataset. From this network we are able to determine the position of the faces.

Then to complete this task we used a [pretrained ResNet18](https://pytorch.org/hub/pytorch_vision_resnet/) network with customized fully convolutional layers. Given the image of a face (obtained from the previous network), the model returns a 2D-tensor with values between 0 and 1. The first value corresponds to the probability of having no mask and the second the probability to have one.

The implementation, the training and the testing of the models are available in the [FaceDetection](./FaceDetection.ipynb) notebook and the [MaskRecognition](./MaskRecognition.ipynb) notebook.

Here are the results obtained on the training datasets and testing datasets respectively:

***TODO: insert both tabs (TP,TN,FN,TN)***

As comparaison to the ResNet18 network we also used a [ResNet50]((https://pytorch.org/hub/pytorch_vision_resnet/)) network and a [VGG19](https://pytorch.org/hub/pytorch_vision_vgg/) network. Similarly as before, here are the results obtained using them:

***TODO: insert both tabs (TP,TN,FN,TN)***

___

### **Task 2 | Human Detection and Tracking**

*Detect humans in videos and track them using bounding boxes over time.*

***TODO: how and results***

- Extras:
    - For human detection, use Yolo (or any other detector that you prefer)
    - Use a SORT tracker to link the human detections. Note: consider using also another tracker for comparison
    - Use mAP for person detection (if you need to train anything)
    - Videos: you can use your own videos (film yourself or download some from YouTube).

___

### **Task 3 | Video Mask Detection: Examine if people in a video are wearing masks**

*Using the results from the Task 1 & task 2, examine if people in videos are wearing masks. Moreover, given that a face is wearing a mask in some frames but not in some other frames, make the prediction unified. Post-process people tracks (merge?, split? delete?)*

***TODO: how and results***

- Extras:
    - Videos: you can use your own videos (film yourself or download some from YouTube).

___

### **Task 4: Social distancing**

*Open problem*

***TODO: Our understanding of the problem***

***TODO: Our solution***

- Extras:
    - For social distancing, you can use anything from heuristics (e.g., use prior on the human height and estimate the distance given this prior), depth estimators (example),

___

### **References**
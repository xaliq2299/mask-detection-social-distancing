# **INF634 - COMPUTER VISION PROJECT**
## **Mask Detection and Social Distancing**


Wearing masks and keeping necessary social distance is essential in the era of COVID.
The goal of this project is to detect if people are wearing masks (in images and over time in
videos) and then compute the distance between them over time. For this, the team will need to
use face detection and tracking techniques, and explore the social distancing effect.

● Dataset with masks: Kaggle link 1 , Kaggle link 2

- Task 1: Mask recognition: classify people wearing masks
    - Train a Mask classification network (binary classification, wearing or not masks)

- Task 2: Human detection and tracking
    - Detect humans in videos   
    - Track human bounding boxes over time

- Task 3: Video mask detection: Examine if people in a video are wearing masks
    - Using the results from Tasks 1 & 2 , examine if people in videos are wearing masks.
    - Data association:
        - Given that a face is wearing a mask in some frames but not in some other
frames, make the prediction unified
        - Post-process people tracks (merge?, split? delete?)

- Task 4: Social distancing:
    - Open problem.

___

- Extras:
    - Use ResNet18 for mask classification (or any other small network that you are familiar
with and you can train in Colab). Note: consider using also another network for
comparison
    - For human detection, use Yolo (or any other detector that you prefer)
    - Use a SORT tracker to link the human detections. Note: consider using also another
tracker for comparison
    - Videos: you can use your own videos (film yourself or download some from YouTube).
    - Metrics:
        - Use classification accuracy for Task 1
        - Use mAP for person detection (if you need to train anything)
    - For social distancing, you can use anything from heuristics (e.g., use prior on the
human height and estimate the distance given this prior), depth estimators (example),
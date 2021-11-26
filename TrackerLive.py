import cv2

import torch
import torch.nn as nn

#################################################################################
# LOAD THE MODELS
#################################################################################



#################################################################################
# LAUNCH THE TRACKING
#################################################################################

cap = cv2.VideoCapture(0)

# check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

_, frame = cap.read()
cv2.imshow("Live Tracker", frame)

while cv2.getWindowProperty("Live Tracker", 0) >= 0:

    _, frame = cap.read()
    cv2.imshow("Live Tracker", frame)
    
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

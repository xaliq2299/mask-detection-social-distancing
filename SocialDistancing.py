import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
import math

class SocialDistancing:

	def __init__(self, min_distance):
		self.min_distance = min_distance
		self.violation_color = (0,0,255)
		self.non_violation_color = (0,255,0)

	def euclidean(self, image, bounding_boxes): # Note: bounding boxes here correspond to the human boxes
	    # TODO: needed the next one?
	    # if bounding_boxes != ():
	    #     centroids = np.empty([bounding_boxes.shape[0], 2]) # needed for social distancing
	    centroids = np.empty([bounding_boxes.shape[0], 2]) # needed for social distancing
	    for i, (x,y,w,h) in enumerate(bounding_boxes):
	        centroids[i] = [x+w//2, y+h//2]

	    D = dist.cdist(centroids, centroids, metric="euclidean")
	    violate = set() # keeps track of violating people saving their indices
	    # loop over the upper triangular of the distance matrix
	    for i in range(0, D.shape[0]):
	        for j in range(i + 1, D.shape[1]):
	            # check to see if the distance between any two centroid pairs is less than the configured number of pixels
	            if D[i, j] < self.min_distance:
	                # update our violation set with the indexes of the centroid pairs
	                violate.add(i)
	                violate.add(j)

	    # print("Violated set:", violate)
	    for i, (x,y,w,h) in enumerate(bounding_boxes):
	        if i in violate:
	            color = self.violation_color
	        else:
	            color = self.non_violation_color
	        cX, cY = centroids[i][0], centroids[i][1]
	        startX, startY, endX, endY = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][0]+bounding_boxes[i][2], bounding_boxes[i][1]+bounding_boxes[i][3]
	        cv.rectangle(image, (startX, startY), (endX, endY), color, 2)
	        # cv.circle(image, (int(cX), int(cY)), 5, color, 2)

	        text = 'Violating people: ' + str(len(violate))
	        cv.putText(image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

	    return image

	def depth(self, image, bounding_boxes, coords):
	    centroids = np.empty([bounding_boxes.shape[0], 2]) # needed for social distancing
	    for i, (x,y,w,h) in enumerate(bounding_boxes):
	        centroids[i] = [x+w//2, y+h//2] # todo: correct?

	    D = np.empty([centroids.shape[0], centroids.shape[0]])
	    violate = set() # keeps track of violating people saving their indices
	    # loop over the upper triangular of the distance matrix
	    for i in range(0, D.shape[0]):
	        for j in range(i + 1, D.shape[1]):
	            x0 = int(coords[int(centroids[i][1]), int(centroids[i][0]), 0])
	            y0 = int(coords[int(centroids[i][1]), int(centroids[i][0]), 1])
	            z0 = int(coords[int(centroids[i][1]), int(centroids[i][0]), 2])
	            x1 = int(coords[int(centroids[j][1]), int(centroids[j][0]), 0])
	            y1 = int(coords[int(centroids[j][1]), int(centroids[j][0]), 1])
	            z1 = int(coords[int(centroids[j][1]), int(centroids[j][0]), 2])
	            D[i, j] = math.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
	            if D[i, j] < self.min_distance:
	                # update our violation set with the indexes of the centroid pairs
	                violate.add(i)
	                violate.add(j)

	    # print("Violated set:", violate)
	    for i, (x,y,w,h) in enumerate(bounding_boxes):
	        if i in violate:
	            color = self.violation_color
	        else:
	            color = self.non_violation_color
	        cX, cY = centroids[i][0], centroids[i][1]
	        startX, startY, endX, endY = bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][0]+bounding_boxes[i][2], bounding_boxes[i][1]+bounding_boxes[i][3]
	        cv.rectangle(image, (startX, startY), (endX, endY), color, 2)
	        # cv.circle(image, (int(cX), int(cY)), 5, color, 2)

	        text = 'Violating people: ' + str(len(violate))
	        cv.putText(image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

	    return image

from tqdm import tqdm
from xml.dom import minidom

import cv2
import numpy as np
import os
import pandas as pd



data_root_path = "./data/"
data_raw_path = data_root_path + "FaceMaskDetection_Raw/"
data_pro_path = data_root_path + "FaceMaskDetection_Processed/"

if "FaceMaskDetection_Processed" not in os.listdir(data_root_path):
    os.mkdir(data_root_path + "FaceMaskDetection_Processed/")
    os.mkdir(data_root_path + "FaceMaskDetection_Processed/images")

images_raw_path = data_raw_path + "images/"
images_pro_path = data_pro_path + "images/"

annotations_raw_path = data_raw_path + "annotations/"
annotations_pro_path = data_pro_path + "annotations/"

images_raw_files = os.listdir(images_raw_path)
annotations_raw_files = os.listdir(annotations_raw_path)

# 1st check-point: same number of files and same file ids in the same order
assert ([annotation_raw_file[15:-4] for annotation_raw_file in annotations_raw_files] == [image_raw_file[15:-4] for image_raw_file in images_raw_files])


name_to_label =  {"without_mask": 0, "mask_weared_incorrect": 1, "with_mask": 2}

nb_images = 0

data = []


for i in tqdm(range(len(images_raw_files))):

    targets = []
    count = 0
    image_id = nb_images
    nb_images += 1

    image_raw_file = images_raw_files[i]
    img = cv2.imread(images_raw_path + image_raw_file)
    cv2.imwrite(images_pro_path + str(image_id) + ".png", img)

    annotation_raw_file = annotations_raw_files[i]
    annotation = minidom.parse(annotations_raw_path + annotation_raw_file)

    image_height = int(annotation.getElementsByTagName("height")[0].firstChild.data)
    image_width = int(annotation.getElementsByTagName("width")[0].firstChild.data)

    for box_id,object in enumerate(annotation.getElementsByTagName("object")):

        box_label = name_to_label[object.getElementsByTagName("name")[0].firstChild.data]
        xmin = int(object.getElementsByTagName("xmin")[0].firstChild.data)
        xmax = int(object.getElementsByTagName("xmax")[0].firstChild.data)
        ymin = int(object.getElementsByTagName("ymin")[0].firstChild.data)
        ymax = int(object.getElementsByTagName("ymax")[0].firstChild.data)

        if box_label < 2:
            targets.append((xmin,xmax,ymin,ymax,box_label))
        else:
            count += 1

        data.append((image_id, image_height, image_width, box_id, box_label, xmin, xmax, ymin, ymax))
    
    if len(targets) > 0:

        img_flip = cv2.flip(cv2.imread(images_raw_path + image_raw_file), 1)

        if len(targets) > count:
            image_id_flip = nb_images
            nb_images += 1
            cv2.imwrite(images_pro_path + str(image_id_flip) + ".png", img_flip)
        
        if  len(targets) > 2*count:
            image_id_flip_bis = nb_images
            nb_images += 1
            cv2.imwrite(images_pro_path + str(image_id_flip_bis) + ".png", img_flip)

        for k in range(len(targets)):

            xmin,xmax,ymin,ymax,box_label = targets[k]

            xmax_flip = min(image_width-1,int(image_width-1-xmin))
            xmin_flip = max(0,int(image_width-1-xmax))
            w = xmax-xmin
            w_flip = xmax_flip-xmin_flip
            h = ymax-ymin

            if len(targets) > count:
                data.append((image_id_flip, image_height, image_width, k, box_label, xmin_flip, xmax_flip, ymin, ymax))
            
            if len(targets) > 2*count:
                data.append((image_id_flip_bis, image_height, image_width, k, box_label, xmin_flip, xmax_flip, ymin, ymax))

            img_bis = img[max(ymin-h,0):min(image_height-1,ymax+h),max(xmin-w,0):min(image_width-1,xmax+w)]
            img_bis_flip = img_flip[max(ymin-h,0):min(image_height-1,ymax+h),max(xmin_flip-w_flip,0):min(image_width-1,xmax_flip+w_flip)]

            cv2.imwrite(images_pro_path + str(nb_images) + ".png", img_bis)
            cv2.imwrite(images_pro_path + str(nb_images+1) + ".png", img_bis_flip)

            h0,w0,_ = img_bis.shape
            h0_flip,w0_flip,_ = img_bis_flip.shape
            xmin = w0//3
            xmax = 2*w0//3
            xmin_flip = w0_flip//3
            xmax_flip = 2*w0_flip//3
            ymin = h0//3
            ymax = 2*h0//3

            data.append((nb_images, h0, w0, 0, box_label, xmin, xmax, ymin, ymax))
            data.append((nb_images+1, h0_flip, w0_flip, 0, box_label, xmin_flip, xmax_flip, ymin, ymax))

            nb_images += 2


columns = ["image_id", "image_height", "image_width", "box_id", "box_label", "xmin", "xmax", "ymin", "ymax"]
annotations = pd.DataFrame(data=data, columns=columns, index=None)
annotations.to_csv(data_pro_path + "annotations.csv", index=None)

print(nb_images)
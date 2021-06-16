import os
import numpy as np
import random
import cv2
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import mrcnn.config
from mrcnn import utils, visualize
from mrcnn.model import MaskRCNN
from pathlib import Path

#  Configuration that will be used by Mask R-CNN library

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6
    
    
# Filter the result of Mask R-CNN to only obtain bounding boxes and class names of objects identified as cars

def getCarBoxes(boxes,class_ids):
    car_boxes = []
    
    # class_id 3/8 corresponds to car/truck objects as per COCO dataset
    for i,box in enumerate(boxes):
        if class_ids[i] in [3,8]:
            car_boxes.append(boxes)
            
    return np.array(car_boxes)

# Root directory of the project
ROOT_DIR = "."

# Local path for logs and trained model.One of the best practices and useful while training the model . 
# Will not be used in this demonstration
MODEL_DIR = os.path.join(ROOT_DIR,"logs")


# Path for saving trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")


# Downloading weights of pre-trained model for COCO dataset from the release 
# Executed for the first time when to store the model weights in this repo

if not os.path.exists(COCO_MODEL_PATH):
   utils.download_trained_weights(COCO_MODEL_PATH)
        

# Directory of images to run detection on
IMAGE_DIR = "images"
# /CNR-EXT_FULL_IMAGE_1000x750"


# Create a Mask R-CNN model in inference mode
model = MaskRCNN(mode='inference', config=MaskRCNNConfig(),model_dir=MODEL_DIR)


# Load pre-trained model, this will load weights of model trained on COCO dataset
%time model.load_weights(COCO_MODEL_PATH,by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the person class, use: class_names.index('person')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Location of parking spaces
parked_car_boxes = None

# Load a random image from images folder
file_names = next(os.walk(IMAGE_DIR))[2]

file_names

# choose the first and 3rd image from a sequence of file_names 
# First image is the reference image of completely occupied parking area. It is used for finding all parking spots

image1 = skimage.io.imread(os.path.join(IMAGE_DIR,"item1.jpg"))
image2 = skimage.io.imread(os.path.join(IMAGE_DIR,"item2.jpg"))


# run detection
# runs the detection pipeline and returns a list of dictionaries, one dict per image.

%time result1 = model.detect([image1])    


%time result2 = model.detect([image2])    

r1 = result1[0]
r2 = result2[0]


visualize.display_instances(image1,r1['rois'],r1['masks'], r1['class_ids'], class_names, r1['scores'])

visualize.display_instances(image2,r2['rois'], r2['masks'], r2['class_ids'], class_names, r2['scores'])

# Filter the results to only get identified cars' bounding boxes

car_boxes1 = getCarBoxes(r1['rois'],r1['class_ids'])
car_boxes2 = getCarBoxes(r2['rois'],r2['class_ids'])

# car_boxes1 and car_boxes2 is an array of length of no. of cars identified with each row having set of 
# 4 coordinates y1, x1, y2, x2 ; points 1 and 2 are opposite vertices of the bounding box.

car_boxes1[0].shape

parking_spaces = car_boxes1[0]

# computinig center locations of each spot
center_locs = []

for spot_coords in parking_spaces:
    center_locs.append([int((spot_coords[1]+spot_coords[3])/2), int((spot_coords[0]+spot_coords[2])/2)])


centers = np.array(center_locs)
    

centers.shape

# Draw each box on the frame

for i,box in enumerate(parking_spaces):
    y1, x1, y2, x2 = box
    cv2.rectangle(image1, (x1, y1), (x2, y2),(0,0,255),2)
    cv2.circle(image1,(centers[i][0],centers[i][1]),2,(0,0,255),2)

# show the image on screen

#cv2.imshow('image', image1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite('./image_result/final_image_1.png', image1)

parking_spaces.shape

car_boxes2.shape

# How much car overlaps with the bounding boxes of parking spaces

overlaps = mrcnn.utils.compute_overlaps(car_boxes2[0],parking_spaces)

overlaps.shape

overlap_prob = overlaps.sum(axis=0)

for i,box in enumerate(parking_spaces):
    y1, x1, y2, x2 = box
    
    if overlap_prob[i] >= 0.5:
        occupancy_status = (0,0,255)
    
    else:
        occupancy_status = (0,255,0)
    cv2.rectangle(image2,(x1,y1), (x2,y2) , occupancy_status ,1)
    cv2.circle(image2,(centers[i][0],centers[i][1]),2,occupancy_status,2)
        


cv2.imwrite('./image_result/final_image_2.png', image2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
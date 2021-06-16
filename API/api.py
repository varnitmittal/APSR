from flask import Flask, make_response,request, render_template, jsonify, send_file
from PIL import Image
import base64
import io
import os
from flask_cors import CORS, cross_origin

import tensorflow as tf
import numpy as np
import cv2

import random
import skimage.io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import mrcnn.config
from mrcnn import utils, visualize
from mrcnn.model import MaskRCNN
from pathlib import Path

execution_path = os.getcwd()

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def loadModel():
    ROOT_DIR = "./API/"
    MODEL_DIR = os.path.join(ROOT_DIR,"logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
            
    IMAGE_DIR = os.path.join(ROOT_DIR,"images")

    model = MaskRCNN(mode='inference', config=MaskRCNNConfig(),model_dir=MODEL_DIR)

    model.load_weights(COCO_MODEL_PATH,by_name=True)

    print("model loaded")
    return model



class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6

def getCarBoxes(boxes,class_ids):
    car_boxes = []
    
    # class_id 3/8 corresponds to car/truck objects as per COCO dataset
    for i,box in enumerate(boxes):
        if class_ids[i] in [3,8]:
            car_boxes.append(boxes)
            
    return np.array(car_boxes)


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

def algorithm(image_1_path, image_2_path):

    image1 = skimage.io.imread(image_1_path)
    image2 = skimage.io.imread(image_2_path)

    result1 = model.detect([image1])    
    result2 = model.detect([image2])    

    r1 = result1[0]
    r2 = result2[0]

    car_boxes1 = getCarBoxes(r1['rois'],r1['class_ids'])
    car_boxes2 = getCarBoxes(r2['rois'],r2['class_ids'])

    parking_spaces = car_boxes1[0]

    center_locs = []

    for spot_coords in parking_spaces:
        center_locs.append([int((spot_coords[1]+spot_coords[3])/2), int((spot_coords[0]+spot_coords[2])/2)])

    centers = np.array(center_locs)
        
    for i,box in enumerate(parking_spaces):
        y1, x1, y2, x2 = box
        cv2.rectangle(image1, (x1, y1), (x2, y2),(0,0,255),2)
        cv2.circle(image1,(centers[i][0],centers[i][1]),2,(0,0,255),2)

    overlaps = mrcnn.utils.compute_overlaps(car_boxes2[0],parking_spaces)

    overlap_prob = overlaps.sum(axis=0)

    for i,box in enumerate(parking_spaces):
        y1, x1, y2, x2 = box
        
        if overlap_prob[i] >= 0.5:
            occupancy_status = (0,0,255)
        
        else:
            occupancy_status = (0,255,0)
        cv2.rectangle(image2,(x1,y1), (x2,y2) , occupancy_status ,1)
        cv2.circle(image2,(centers[i][0],centers[i][1]),2,occupancy_status,2)
            
    cv2.imwrite("./API/image_result/final_image.png", image2)
    return true


def encodeFinal(finalImagePath):
    with open(finalImagePath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string


@app.route("/", methods=['GET'])
@cross_origin()
def index():
    r = make_response(render_template("home.html"))
    r.headers.set("Access-Control-Allow-Origin", "*")
    return r


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    """ content = request.get_json()
    image_in = content['chosenImage']
    image_in = base64.b64decode(image_in)
    image_in = Image.open(io.BytesIO(image_in))
    if image_in.mode != "RGB":
        image_in = image_in.convert("RGB")
    image_in.save(r'received_image.png') """

    #image_1_path = os.path.join(IMAGE_DIR,"item1.jpg")    
    #image_2_path = os.path.join(IMAGE_DIR,"/API/received_image.png")
    image_1_path = './API/item1.jpg'
    image_2_path = './API/received_image.png'

    flag = algorithm(image_1_path, image_2_path)

    if flag == False:
        response = jsonify({
            "success": False
        })
        return response
    else: 
        final_image_path = './API/image_result/final_image.png'
        response = jsonify({
            "success": True,
            #"finalImage": str(encodeFinal(final_image_path))
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        return response


#get_model()
if __name__ == "__main__":
    model = loadModel()
    port = int(os.environ.get('PORT', 8000))
    app.run(port=port)
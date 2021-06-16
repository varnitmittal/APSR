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

ROOT_DIR = "./API/"
MODEL_DIR = os.path.join(ROOT_DIR,"logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
   utils.download_trained_weights(COCO_MODEL_PATH)
        
IMAGE_DIR = os.path.join(ROOT_DIR,"images")

model = MaskRCNN(mode='inference', config=MaskRCNNConfig(),model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH,by_name=True)

print("model loaded")

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

def algorithm():
    image1 = skimage.io.imread(os.path.join(IMAGE_DIR,"item1.jpg"))
    image2 = skimage.io.imread(os.path.join(IMAGE_DIR,"item2.jpg"))

    def processIm1():
        result1 = model.detect([image1])    
        print("image1 prediction ready")
        return result1[0]

    def processIm2():
        result2 = model.detect([image2])    
        print("image2 prediction ready")
        return result2[0]

    r1 = processIm1()
    r2 = processIm2()

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
    print("final image written")

    #image2 = cv2.imread("./API/image_result/final_image.png", cv2.IMREAD_UNCHANGED)
    with open("./API/image_result/final_image.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string

















""" 
def loadModel(jsonStr, weightStr):
    json_file = open(jsonStr, 'r')
    loaded_nnet = json_file.read()
    json_file.close()

    serve_model = tf.keras.models.model_from_json(loaded_nnet)
    serve_model.load_weights(weightStr)

    serve_model.compile(optimizer=tf.optimizers.Adam(),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    return serve_model

model = loadModel('model.json', 'model.h5')



# Match contours to license plate or character template
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        #checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            #plt.imshow(ii, cmap='gray')

            #Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) #List that stores the character's binary image (unsorted)
            
    #Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    #plt.show()
    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

# Find characters in the resulting images
def segment_characters(image) :
    
    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    #plt.imshow(img_binary_lp, cmap='gray')
    #plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

#Inference
def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28))
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        y_ = model.predict_classes(img)[0] #predicting the class
        character = dic[y_] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number
 """
@app.route("/", methods=['GET'])
@cross_origin()
def index():
    r = make_response(render_template("home.html"))
    r.headers.set("Access-Control-Allow-Origin", "*")
    return r


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    content = request.get_json()
    image_in = content['chosenImage']
    image_in = base64.b64decode(image_in)
    image_in = Image.open(io.BytesIO(image_in))
    if image_in.mode != "RGB":
        image_in = image_in.convert("RGB")
    image_in.save(r'received_image.png')
    img = cv2.imread('received_image.png', cv2.IMREAD_UNCHANGED)
    ########
    char = segment_characters(img)
    plate = show_results(char)
    print("Working till here!")
    response = jsonify({
        "success": True,
        "imstr": "123456789",
        "detections": plate
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    return response


#get_model()
if __name__ == "__main__":

    port = int(os.environ.get('PORT', 8000))
    app.run(port=port)
'''
    Author: Jordan Madden
    Usage: python object_detection.py --image=images/example_01.jpg
           python object_detection.py --image=images/example_02.jpg
           python object_detection.py --image=images/example_03.jpg
           python object_detection.py --image=images/example_04.jpg
           python object_detection.py --image=images/example_05.jpg 
           python object_detection.py --image=images/example_06.jpg
'''

import numpy as np
import argparse
import time
import cv2

# Parse the command line arguements
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Declare relevant constants
PROTOTXT = "caffe_models/MobileNetSSD_deploy.prototxt.txt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# Initialize the list containing the names of classes and another to store
# the color that will be used to represent each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the model from disk
print("[INFO] Loading model...")
detector = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Load and preprocess the image before passing it to the detector
image = cv2.imread(args["image"])
H, W = image.shape[:2]
imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
    (300, 300), 127.5)

# Pass the blob through the network and obtain the predictions
print("[INFO] Computing detections")
startTime = time.time()
detector.setInput(imageBlob)
detections = detector.forward()
print("[INFO] The image was processed in {:.2f} seconds".format(time.time()-startTime))

for i in np.arange(0, detections.shape[2]):
    # Extract the confidence from the detections
    confidence = detections[0, 0, i, 2]

    # Filter out the weak detections
    if confidence >= args["confidence"]:
        # Extract the index of the class label, then find the coordinates
        # of the bounding boxes
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7]*np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")

        # Display the predictions
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
        cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
        	cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# Show the output image
cv2.imshow("Image", image)
cv2.waitKey()
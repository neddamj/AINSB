'''
    Author: Jordan Madden
    Usage: python recognize.py 
'''

import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Declare relevant constants
IMAGE = "images/elspeth_madden.jpg"
DETECTOR = "face_detection_model"
EMBEDDER = "openface_nn4.small2.v1.t7"
RECOGNIZER = "output/recognizer.pickle"
LE = "output/le.pickle"
CONFIDENCE = 0.5

# Load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
modelPath = os.path.sep.join([DETECTOR,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(EMBEDDER)

# Load the actual face recognition model and the label encoder
recognizer = pickle.loads(open(RECOGNIZER, "rb").read())
le = pickle.loads(open(LE, "rb").read())

# Load and preprocess the image, then grab the image dimensions
image = cv2.imread(IMAGE)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# Construct a blob from the image and detect faces in the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)
detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
	# Find the confidence of the predictions
	confidence = detections[0, 0, i, 2]

	# Filter out weak detections
	if confidence > CONFIDENCE:
		# Compute the bounding box coordinates of the face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		if fW < 20 or fH < 20:
			continue

		# Preprocess the face, then extract the face embeddings
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# Recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# Draw the bounding box of the face along with the associated
		# probability
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
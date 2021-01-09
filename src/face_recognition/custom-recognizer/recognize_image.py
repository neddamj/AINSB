'''
	Author: Jordan Madden
	Title: recognize_image.py
	Date: 9/1/2021
	Usage:python recognize_image.py 
'''

import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# declare relevant constants(filepaths etc)
IMAGE = "images/jordan_madden.jpg"
DETECTOR = "face_detection_model"
EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
RECOGNIZER = "output/recognizer.pickle"
CONFIDENCE = 0.50
LE = "output/le.pickle"

# load the serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
modelPath = os.path.sep.join([DETECTOR,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(RECOGNIZER, "rb").read())
le = pickle.loads(open(LE, "rb").read())

# load and resize the image, then get the new image dimensions
image = cv2.imread(IMAGE)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply the face detection model to find faces
detector.setInput(imageBlob)
detections = detector.forward()

for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > CONFIDENCE:
		# find the (x, y)-coordinates of the bounding box for the face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		# construct a blob for the face ROI, then run it through the facial embeddings model
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# draw the bounding box of the face along with the associated
		# probability
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
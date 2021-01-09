'''
	Author: Jordan Madden
	Title: recognize_video.py
	Date: 9/1/2021
	Usage: python recognize_video.py 
'''

# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# declare relevant constants(filepaths etc)
DETECTOR = "face_detection_model"
EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
RECOGNIZER = "output/recognizer.pickle"
CONFIDENCE = 0.50
LE = "output/le.pickle"

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
modelPath = os.path.sep.join([DETECTOR,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(RECOGNIZER, "rb").read())
le = pickle.loads(open(LE, "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
	# grab the frame from the threaded video stream
	ret, frame = cap.read()
	if not ret:
		break

	# resize the frame to have a width of 600 pixels and get the frame dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image and pass it through the face detector
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence >= CONFIDENCE:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then run it through the facial embeddings model
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face and its probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

#End the video stream
cap.release()
cv2.destroyAllWindows()
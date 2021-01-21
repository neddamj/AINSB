'''
    Author: Jordan Madden
    Usage: python train_recognizer.py
'''

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Declare relevant constants
DATASET = "dataset"
DETECTOR = "face_detection_model"
EMBEDDER = "openface_nn4.small2.v1.t7"
RECOGNIZER = "output/recognizer.pickle"
EMBEDDINGS = "output/embeddings.pickle"
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

# Get the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(DATASET))

# Create the lists for the face embeddings and corresponding names
knownEmbeddings = []
knownNames = []

# Initialize the total number of faces processed
total = 0

for (i, imagePath) in enumerate(imagePaths):
	# Extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# Load and preprocess the image, then grab the image dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# Construct a blob from the image and detect faces in the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()

	if len(detections) > 0:
		# Find the confidence of each detection
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# Filter out weak detections
		if confidence > CONFIDENCE:
			# Compute the bounding box coordinates of the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:
				continue

			# Preprocess the face, then extract the face embeddings
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# Add the name of the person + corresponding face
			# embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# Dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(EMBEDDINGS, "wb")
f.write(pickle.dumps(data))
f.close()

# encode the labels
print("[INFO] encoding labels...", end="")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
print("DONE")

# Train the model used to actually recognize the faces
print("[INFO] training model...", end="")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)
print("DONE")

# Write the actual face recognition model to disk
f = open(RECOGNIZER, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Write the label encoder to disk
f = open(LE, "wb")
f.write(pickle.dumps(le))
f.close()
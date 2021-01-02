'''
Author: Jordan Madden
Usage:python train_model.py --dataset=dataset --embeddings=output/embeddings.pickle --detector=face_detection_model --embedding-model=openface_nn4.small2.v1.t7 --recognizer=output/recognizer.pickle --le=output/le.pickle
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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")	
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

'''
Section 1: Extracting the facial embeddings
'''

# load the serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# grab the paths to the input images in our dataset and initialize lists containing
#facial embeddings and names
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
num_faces = 0

for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load and resize the image, then get the new image dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply the face detection model to find faces
	detector.setInput(imageBlob)
	detections = detector.forward()

	if len(detections) > 0:
		# assuming each image has 1 face, find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then run it through the 
			#facial embeddings model
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# populate the lists of embeddings and names
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			num_faces += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(num_faces))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()

'''
Section 2: Classifying the faces based on the extracted embeddings
'''

# encode the labels
print("[INFO] encoding labels...", end="")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
print("DONE")

# train the model that classifies the face based on the embeddings
print("[INFO] training model...", end="")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)
print("DONE")

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
'''
	Author: Jordan Madden
	Title: train_model.py
	Date: 9/1/2021
	Usage: python train_model.py
'''

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# declare relevant constants(filepaths etc)
DATASET = "dataset"
EMBEDDINGS = 'output/embeddings.pickle'
DETECTOR = "face_detection_model"
EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
RECOGNIZER = "output/recognizer.pickle"
CONFIDENCE = 0.5
LE = "output/le.pickle"

'''
	Section 1: Extracting the facial embeddings
'''

# load the serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
modelPath = os.path.sep.join([DETECTOR,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)

# grab the paths to the input images in our dataset and initialize lists containing
#facial embeddings and names
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(DATASET))
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
		if confidence > CONFIDENCE:
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
print("[INFO] serializing {} embeddings...".format(num_faces))
data = {"embeddings": knownEmbeddings, 
		"names": knownNames}
f = open(EMBEDDINGS, "wb")
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
f = open(RECOGNIZER, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(LE, "wb")
f.write(pickle.dumps(le))
f.close()
'''
	Author: Jordan Madden
	Title: encode_faces.py
	Date: 9/1/2021
'''

from imutils import paths
import face_recognition
import pickle
import cv2
import os

#Declare the relevant constants(filepaths etc)
DATASET = "dataset"
ENCODINGS = "encodings.pickle"
DETECTION_METHOD = "hog"

# grab the paths to the input images in our dataset and initialize the list of known encodings and names
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(DATASET))
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from BGR to RGB
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes for the faces and generate the facial embeddings
	boxes = face_recognition.face_locations(rgb,
		model=DETECTION_METHOD)
	encodings = face_recognition.face_encodings(rgb, boxes)

	# add each encoding and name to the list of known names and encodings
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(ENCODINGS, "wb")
f.write(pickle.dumps(data))
f.close()
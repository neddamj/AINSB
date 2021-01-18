'''
    Author: Jordan Madden
    Usage: python face_recognition.py
    Purpose: This module aims to compile the functionalities necessary for face recognition
             into an easy to use class and provide sample code that shows how to use the 
             module.
    Project Structure: To use this module, the following folders should be present in your working directory:
                        -dataset
                        -face_detection_model
                        -images
                        The "dataset" folder should contain the folders with the images of the people that you want 
                        the system to recognize. The folder containing the images of a person should be titled as 
                        the name of the person you want the system to recognize. There should be also be a folder 
                        inside of "dataset" called Unknown which contains images of arbitrary faces.

                        The "face_detection_model" folder should contain the .prototxt file and the .caffemodel file
                        of the pre-trained Caffe face detector.

                        The "images" folder should contain images with the faces of people that you wish to test the 
                        system on.
'''

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

class FaceRecognition:
    def __init__(self, dataset="dataset", output_path='output', detector="face_detection_model", confidence=0.5):
        # Declaring the relevant constants 
        self.DATASET = dataset
        self.OUTPUT_PATH = "output"
        self.DETECTOR = detector
        self.CONFIDENCE = confidence
        self.EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
        self.EMBEDDINGS = os.path.join(self.OUTPUT_PATH, 'embeddings.pickle')
        self.RECOGNIZER = os.path.join(self.OUTPUT_PATH, "recognizer.pickle")
        self.LE = os.path.join(self.OUTPUT_PATH, "le.pickle")

        # If the "output" directory is not present in the working directory,
        # then create it
        if not os.path.exists(self.OUTPUT_PATH):
            os.mkdir(self.OUTPUT_PATH)

    def load_detector(self):
        '''
            Inputs: Nothing
            Purpose: Load the pre-trained Caffe face detection model for use.
            Returns: Pre-trained Caffe face detector
        '''
        # load the serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.sep.join([self.DETECTOR, "deploy.prototxt"])
        modelPath = os.path.sep.join([self.DETECTOR,
            "res10_300x300_ssd_iter_140000.caffemodel"])
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        return detector

    def load_embedder(self):
        '''
            Inputs: Nothing
            Purpose: Load the pre-trained FaceNet face embedding model for use.
            Returns: Pre-trained FaceNet face embedding model
        '''
        # load the serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch(self.EMBEDDING_MODEL)

        return embedder

    def save_embeddings(self, num_faces, knownEmbeddings, knownNames):
        '''
            Inputs: The number of faces detected, the extracted face embeddings, 
                    the names that were extracted with each embedding
            Purpose: Save the extracted facial embeddings and their corresponding 
                     names to a .pickle file for future use.
            Returns: Nothing
        '''
        # dump the facial embeddings + names to disk
        print("[INFO] serializing {} embeddings...".format(num_faces))
        data = {"embeddings": knownEmbeddings, 
                "names": knownNames}
        f = open(self.EMBEDDINGS, "wb")
        f.write(pickle.dumps(data))
        f.close()

    def detect_faces(self, image, detector):
        '''
            Inputs: The image that the model is to be run on, the pre-trained 
                    Caffe model.
            Purpose: Load the pre-trained FaceNet face embedding model for use.
            Returns: The detections of each face in the image
        '''
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply the face detection model to find faces
        detector.setInput(imageBlob)
        detections = detector.forward()

        return detections

    def get_embeddings(self, face, embedder):
        '''
            Inputs: The face that is to be recognized, the pre-trained FaceNet 
                    facial embeddings model
            Purpose: Computes the embeddings of the face
            Returns: The embeddings of the face 
        '''        
        # construct a blob for the face ROI, then run it through the 
        # facial embeddings model
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        return vec

    def load_embeddings(self):
        '''
            Inputs: Nothing
            Purpose: Load the contents of the .pickle file that contains the saved 
                     face embeddings
            Returns: A dictionary containing the face embeddings and their corresponding
                     names
        '''
        # Open the file containing the facial embeddings
        data = pickle.loads(open(self.EMBEDDINGS, "rb").read())

        return data

    def load_labels_and_recognizer(self, recognizer, le):
        '''
            Inputs: The face recognition SVM model, the label encoder
            Purpose: Load the face recognition SVM and the label encoder from the 
                     pickle files they were saved in on the disk
            Returns: The recognition SVM and the label encoder
        '''        
        # load the actual face recognition model along with the label encoder
        recognizer = pickle.loads(open(recognizer, "rb").read())
        le = pickle.loads(open(le, "rb").read())

        return recognizer, le

    def save_models(self, recognizer, le):
        '''
            Inputs: The image that the model is to be run on, the pre-trained 
                    Caffe model.
            Purpose: Load the pre-trained FaceNet face embedding model for use.
            Returns: The detections of each face in the image
        '''            
        # write the actual face recognition model to disk
        f = open(RECOGNIZER, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        f = open(LE, "wb")
        f.write(pickle.dumps(le))
        f.close()

    def get_bb_coordinates(self, detections, h, w, i):
        '''
            Inputs: The face detections from the pre-trained Caffe model, the height 
                    and width of the face and the index of the loop that the function 
                    is called in.
            Purpose: To extract the coordinates of the face in the image
            Returns: A tuple containing the coordinates of the bounding box of the face
        '''
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        return (startX, startY, endX, endY)

    def extract_embeddings(self):
        '''
            Inputs: Nothing
            Purpose: To extract the embeddings of the faces in the training dataset
            Returns: Nothing
        '''        
        # Load the face detection model and the embedder model
        detector = self.load_detector()
        embedder = self.load_embedder()

        # Grab the paths to the input images in our dataset and initialize lists containing
        # facial embeddings and names
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.DATASET))
        knownEmbeddings = []
        knownNames = []

        # Initialize the total number of faces processed
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

            # Detect faces in the image
            detections = self.detect_faces(image, detector)

            if len(detections) > 0:
                # assuming each image has 1 face, find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    (startX, startY, endX, endY) = self.get_bb_coordinates(detections, h, w, i)

                    # extract the face ROI and grab the ROI dimensions
                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]
                    if fW < 20 or fH < 20:
                        continue
                    
                    # Get the facial embeddings
                    vec = self.get_embeddings(face, embedder)

                    # populate the lists of embeddings and names
                    knownNames.append(name)
                    knownEmbeddings.append(vec.flatten())
                    num_faces += 1

        # Save the embeddings and their corresponding names to memory
        self.save_embeddings(num_faces, knownEmbeddings, knownNames)

    def train_classifier(self):
        '''
            Inputs: Nothing
            Purpose: Train the SVM that will be used to classify the extracted embeddings 
                     of each face
            Returns: Nothing
        '''        
        # Load the facial embeddings
        data = self.load_embeddings()

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
        self.save_models(recognizer, le)
        
    def classify_face(self, recognizer, vec):
        '''
            Inputs: The SVM classification model that will be used to classify the 
                    face, the extracted embeddings of a face
            Purpose: Determine the name of a person and the probability that it is them
            Returns: The name of the person in the image, the probability that the 
                     embedding represents that person
        '''        
        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        return name, proba

    def train_recognizer(self):
        '''
            Inputs: Nothing
            Purpose: To train the face recognition system
            Returns: Nothing
        '''        
        self.extract_embeddings()
        self.train_classifier()


if __name__ == "__main__":
    # declare relevant constants(filepaths etc)
    IMAGE = "images/elspeth_madden.jpg"
    DETECTOR = "face_detection_model"
    EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
    RECOGNIZER = "output/recognizer.pickle"
    CONFIDENCE = 0.50
    LE = "output/le.pickle"
        
    # Create the face recognition object
    rec = FaceRecognition()

    # Train the face recognition model
    #rec.train_recognizer()
   
    # Load the face detector and the embeddings model from disk
    detector = rec.load_detector()
    embedder = rec.load_embedder()

    # Load the actual face recognition model and the label encoder
    recognizer, le = rec.load_labels_and_recognizer(RECOGNIZER, LE)

    # load and resize the image, then get the new image dimensions
    image = cv2.imread(IMAGE)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Find the faces in the image
    detections = rec.detect_faces(image, detector)    

    for i in range(0, detections.shape[2]):
        # Extract the confidence of each detection
        confidence = detections[0, 0, i, 2]

        if confidence >= CONFIDENCE:
            # Find the bounding box coordinates
            (startX, startY, endX, endY) = rec.get_bb_coordinates(detections, h, w, i)

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # Get facial embeddings in image
            vec = rec.get_embeddings(face, embedder)

            # Classify the face of the person
            name, prob = rec.classify_face(recognizer, vec)

            # draw the bounding box of the face along with the associated
            # probability
            text = "{}: {:.2f}%".format(name, prob * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            cv2.imshow("Image", image)
            cv2.waitKey()



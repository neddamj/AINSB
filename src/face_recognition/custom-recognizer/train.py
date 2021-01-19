from face_recognition import FaceRecognition
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

if __name__ == "__main__":  
    # declare relevant constants(filepaths etc)
    DATASET = "dataset"
    DETECTOR = "face_detection_model"
    EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
    RECOGNIZER = "output/recognizer.pickle"
    CONFIDENCE = 0.50
    LE = "output/le.pickle"

    # Create face rec object
    rec = FaceRecognition()

    # Load the face detection model and the embedder model
    detector = rec.load_detector()
    embedder = rec.load_embedder()

    # Grab the paths to the input images in our dataset and initialize lists containing
    # facial embeddings and names
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(DATASET))
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
        detections = rec.detect_faces(image, detector)

        if len(detections) > 0:
            # assuming each image has 1 face, find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                (startX, startY, endX, endY) = rec.get_bb_coordinates(detections, h, w, i)

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                    
                # Get the facial embeddings
                vec = rec.get_embeddings(face, embedder)

                # populate the lists of embeddings and names
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                num_faces += 1

    # Save the embeddings and their corresponding names to memory
    rec.save_embeddings(num_faces, knownEmbeddings, knownNames)
               
    # Load the facial embeddings
    data = rec.load_embeddings()

    # Train the classifier model
    recognizer, le = rec.train_classifier(data)

    # write the actual face recognition model to disk
    f = open(RECOGNIZER, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(LE, "wb")
    f.write(pickle.dumps(le))
    f.close()
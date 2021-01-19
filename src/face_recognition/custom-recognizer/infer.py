from face_recognition import FaceRecognition
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

if __name__ == "__main__":
    # declare relevant constants(filepaths etc)
    IMAGE = "images/jordan_madden.jpg"
    DETECTOR = "face_detection_model"
    EMBEDDING_MODEL = "openface_nn4.small2.v1.t7"
    RECOGNIZER = "output/recognizer.pickle"
    CONFIDENCE = 0.50
    LE = "output/le.pickle"
        
    # Create the face recognition object
    rec = FaceRecognition()
    
    # Load the face detector and the embeddings model from disk
    detector = rec.load_detector()
    embedder = rec.load_embedder()

    # Load the actual face recognition model and the label encoder
    recognizer = pickle.loads(open(RECOGNIZER, "rb").read())
    le = pickle.loads(open(LE, "rb").read())

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
            name, prob = rec.classify_face(recognizer, vec, le)

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


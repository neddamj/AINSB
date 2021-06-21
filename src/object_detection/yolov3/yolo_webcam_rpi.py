# python yolo_webcam_rpi.py

# import the necessary packages
from realsense import Realsense
import numpy as np
import argparse
import imutils
import time
import cv2
import os

#Declare constants that will be used 
YOLO = "yolo-coco/"
CONFIDENCE = 0.5
THRESHOLD = 0.3

#Load the class labels and the model intializations
labelPath = os.path.join(YOLO, "coco.names")
LABELS = open(labelPath).read().strip().split("\n")
weightsPath = os.path.join(YOLO, "yolov3.weights")
configPath = os.path.join(YOLO, "yolov3.cfg")

#Colors to represnt class labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="int")

print("[INFO]Loading network from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video stream and the FPS counter
print('[INFO] running inference for realsense camera...')
video = RealSense(width=640, height=480).start()

while True:
    # Get the video frames from the camera
    color_frame, depth_frame = video.read()

    # Convert images to numpy arrays and get the frame dimensions
    depth_image = np.asanyarray(depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame, axis=0)

    #Get the frame dimensions
    (H, W) = frame.shape[:2]

    # Record time for a forward pass
    start = time.time()

    #Preprocess the frame and pass it through the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                         swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    print("[INFO] Forward pass takes {:.2f}...".format(time.time() - start))

    #Lists for model output parameters
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            #Extract the class ID and confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4]*np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #Apply non-maxima supression to suppress weak predictions
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    if len(idxs) > 0:
        #Loop over the indexes 
        for i in idxs.flatten():
            #Extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            #Draw bounding box and label frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            label = "{}: {:.2f}%".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()














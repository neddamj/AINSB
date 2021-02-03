'''
    Author: Jordan Madden
    Usage: python TFLite-PiCamera-od.py
'''

import tflite_runtime.interpreter as tflite
import pyrealsense2.pyrealsense2 as rs
from imutils.video import FPS
from threading import Thread
import importlib.util
import numpy as np
import argparse
import time
import cv2
import os
import sys

# Construct and parse the command line arguments 
ap = argparse.ArgumentParser()
ap.add_argument('--model', help='Provide the path to the TFLite file, default is models/model.tflite',
                    default='/home/pi/tflite/detect.tflite')
ap.add_argument('--labels', help='Provide the path to the Labels, default is models/labels.txt',
                    default='/home/pi/tflite/labels.txt')
ap.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
ap.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')               
args = vars(ap.parse_args())

class RealSenseVideo:
    def __init__(self, width=640, height=480):
        # Frame dimensions of camera
        self.width = width
        self.height = height

        # Build and enable the depth and color frames
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
        
        # Read the first frame from the stream
        self.frame = self.pipeline.wait_for_frames()
        self.depth_frame = self.frame.get_depth_frame()
        self.color_frame = self.frame.get_color_frame()

        # Variable to check if thread should be stopped
        self.stopped = False

    def start(self):
        # Start the thread to read the frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            # Stop streaming in indicator is set
            if self.stopped:
                return

            # Otherwise read the next frame in the stream
            self.frame = self.pipeline.wait_for_frames()
            self.depth_frame = self.frame.get_depth_frame()
            self.color_frame = self.frame.get_color_frame()
            if not self.depth_frame or not self.color_frame:
                return

    def read(self):
        # Return the most recent color and depth frame
        return self.color_frame, self.depth_frame

    def stop(self)        :
        # Stop the video stream
        self.stopped = True
        self.pipeline.stop()

def detect(input_data, input_details, output_details):
    # Perform the object detection and get the results
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] 
    classes = interpreter.get_tensor(output_details[1]['index'])[0] 
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    return (boxes, classes, scores)

def visualize_boxes(frame, boxes, scores, classes, H, W):
    # Get the bounding box coordinates
    coordinates = get_bb_coordinates(boxes, scores, H, W)
    i = 0
    
    for coordinate in coordinates:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = coordinate

        # Draw bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

def filter_distance(depth_frame, x, y):
    #List to store the consecutive distance values and randomly initialized variable
    distances = []
    positive = np.random.randint(low=30, high=100)

    i = 0
    while(i < 75):
        # Extract the depth value from the camera
        dist = int(depth_frame.get_distance(x, y) * 100)
        
        # Store the last positive value for use in the event that the
        # value returned is 0
        if dist != 0:
            positive = dist
        elif dist == 0:
            positive == positive

        # Add the distances to the list
        distances.append(positive)
        i += 1

    # Convert the list to a numpy array and return it
    distances = np.asarray(distances)
    return int(distances.mean())

def get_bb_coordinates(detections, scores, H, W, confidence=0.5):
    # Initialize list to store bounding box coordinates of each bounding box
    coordinates = []

    for detection, score in zip(detections, scores):
        # Only move forward if score is above the threshold
        if score > confidence:
            # Extract the coordinates of the detections and normalize each detection
            y1, x1, y2, x2 = detection
            y1 = int(H*y1)
            x1 = int(W*x1)
            y2 = int(H*y2)
            x2 = int(W*x2)

            # Add the coordinates to the coordinate list
            coordinates.append([x1, y1, x2, y2])

    return coordinates

# Declare relevant constants
PATH_TO_MODEL_DIR = args["model"]
PATH_TO_LABELS = args["labels"]
MIN_CONF_THRESH = args["threshold"]

# Get the desired image dimensions
resW, resH = args["resolution"].split('x')
imW, imH = int(resW), int(resH)

# Load TF Lite model
print('[INFO] loading model...')
start_time = time.time()

interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL_DIR)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize video stream and the FPS counter
print('[INFO] running inference for realsense camera...')
video = RealSenseVideo(width=imW, height=imH).start()
fps = FPS().start()
time.sleep(1)

while True:
    # Get the video frames from the camera
    color_frame, depth_frame = video.read()
    
    # Convert images to numpy arrays and get the frame dimensions
    depth_image = np.asanyarray(depth_frame.get_data())
    frame1 = np.asanyarray(color_frame.get_data())

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model(non-quantized model)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
    # Run the object detection and get the results
    boxes, classes, scores = detect(input_data, input_details, output_details)
     
    # Visualize the detections
    visualize_boxes(frame, boxes, scores, classes, imH, imW)
            
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object Detector', frame)

    # Update FPS counter
    fps.update()

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        fps.stop()
        break

# Show the elapsed time and the respective fps
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approximate fps: {:.2f}".format(fps.fps()))
    
# Stop the video stream
cv2.destroyAllWindows()
video.stop()
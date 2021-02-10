'''
    Author: Jordan Madden
    Usage: python test.py --model="ssdmobilenet_v2"
           python test.py --model="efficientdet_d0" 
'''

from playsound import playsound 
from threading import Thread
import pyrealsense2 as rs
import numpy as np
import argparse
import time
import cv2
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    

import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils

# Suppress TensorFlow logging (2)
tf.get_logger().setLevel('ERROR')

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

def model_name(model):
    # Return the name of the model that was specified through the command
    # line arguement
    if model == 'ssdmobilenet_v2':
        return 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    elif model == 'efficientdet_d0':
        return "efficientdet_d0_coco17_tpu-32"

def path_to_ckpt(model):
    # Return the path to the model that was specified through the command
    # line arguement
    if model == 'ssdmobilenet_v2':
        return os.path.join(MODELS_DIR, os.path.join('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'checkpoint/'))
    elif model == 'efficientdet_d0':
        return os.path.join(MODELS_DIR, os.path.join('efficientdet_d0_coco17_tpu-32', 'checkpoint/'))

def path_to_cfg(model):
    # Return the path to the model that was specified through the command
    # line arguement
    if model == 'ssdmobilenet_v2':
        return os.path.join(MODELS_DIR, os.path.join('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config'))
    elif model == 'efficientdet_d0':
        return os.path.join(MODELS_DIR, os.path.join('efficientdet_d0_coco17_tpu-32', 'pipeline.config')) 

@tf.function
def detect(img):
    # Preprocess the image and get the bounding box detections for objects 
    # in the image
    img, shapes = detector.preprocess(img)
    prediction_dict = detector.predict(img, shapes)
    detections = detector.postprocess(prediction_dict, shapes)

    return (detections, prediction_dict, tf.reshape(shapes, [-1]))

def playback(commnds, motion_command):
    #Play audio recording of the given command
    playsound(commands[motion_command])

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
    # Initialize list to store midpoints of each bounding box
    coords = []

    for detection, score in zip(detections, scores):
        # Only move forward if score is above the threshold
        if score > confidence:
            # Extract the coordinates of the detections and normalize each detection
            y1, x1, y2, x2 = detection
            y1 = int(H*y1)
            x1 = int(W*x1)
            y2 = int(H*y2)
            x2 = int(W*x2)

            print("DETECTION")

            # Add the midpoints to the midpoints list
            coords.append([x1, y1, x2, y2])

    return coords

def command(val, frame):
    # Display the command on the screen
    text = "Command: {}".format(val)
    cv2.rectangle(frame, (0, 0), (180, 25), (255, 255, 255), -1)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), thickness=2)

def checkpoints(depth_frame):
    global checkpoint_detection
    checkpoint_detection = False
    W, H = 640, 480

    # Coordinates of the points to be checked in the frame
    center = filter_distance(depth_frame, W//2, H//2)
    right = filter_distance(depth_frame, W//2 + 90, H//2)
    left = filter_distance(depth_frame, W//2 - 90, H//2)
    l_center = filter_distance(depth_frame, W//2, H//2 + 180)
    l_right = filter_distance(depth_frame, W//2 + 60, H//2 + 180)
    l_left = filter_distance(depth_frame, W//2 - 60, H//2 + 180)
    
    # If any of the checkpoints are triggered raise a notification
    if ((center < min_distance) or (left < min_distance) or (right < min_distance) or 
        (l_center < min_distance) or (l_left < min_distance) or (l_right < min_distance)):
        checkpoint_detection = True
        return True
    
    return False

def stop_moving(dist, depth_frame):
    # Stop moving if an object is detected within 1.5 meters or if any of the 
    # chekpoints are triggered
    if (checkpoints(depth_frame)  or (dist < min_distance)):
        return True    
    
    # If none of the conditions are met, keep moving
    return False

def navigate(frame, depth_frame, dist, left, right):
    # Determine the midpoint of each detection and the distance between the object and 
    # the left and right borders of the frame
    midX = (left+right)//2
    dist_left = left - 0
    dist_right = 640 - right
    
    if stop_moving(dist, depth_frame):
        # Stop moving for a bit while deciding what action to take
        command("Stop", frame)
        time.sleep(0.5)

        if dist_left > dist_right:
            command("Left", frame)
        elif dist_right > dist_left:
            command("Right", frame)
    else:
        # Move forward
        command("Forward", frame)

if __name__ == "__main__":
    # Construct and parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="type of model to use")
    args = vars(ap.parse_args())

    # Declare the filepaths and data structures for text 
    # to speech
    COMMAND_PATH = 'text_to_speech/commands'
    FWD = 'forward_command.mp3'
    LEFT = 'left_command.mp3'
    RIGHT = 'right_command.mp3'
    STOP = 'stop_command.mp3'
    forwardPath = os.path.join(COMMAND_PATH, FWD)
    leftPath = os.path.join(COMMAND_PATH, LEFT)
    rightPath = os.path.join(COMMAND_PATH, RIGHT)
    stopPath = os.path.join(COMMAND_PATH, STOP)

    commands = {}
    commands["Forward"] = forwardPath
    commands["Left"] = leftPath
    commands["Right"] = rightPath
    commands["Stop"] = stopPath

    # Declare the relevant constants for object detection
    OD_BASE_PATH = 'object_detection\\tf2'
    DATA_DIR = os.path.join(OD_BASE_PATH, 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    MODEL_NAME = model_name(args["model"])
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
    PATH_TO_CKPT = path_to_ckpt(args["model"])
    PATH_TO_CFG = path_to_cfg(args["model"])

    # Declare the relevant constants for the use of the realsense camera
    SCALE_H = 0.5
    SCALE_W = 0.5

    # Declare variables and constants for navigation
    checkpoint_detection = False
    min_distance = 120
    
    # Build the object detector, restore its weights from the checkpoint file
    # and load the label map
    print("[INFO] building model pipeline and detector...")
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detector = model_builder.build(model_config=model_config, is_training=False)

    print("[INFO] restoring model checkpoint...")
    PATH_TO_RESTORE = os.path.join(PATH_TO_CKPT, 'ckpt-0')
    ckpt = tf.compat.v2.train.Checkpoint(model=detector)
    ckpt.restore(PATH_TO_RESTORE).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # Start the video stream
    print("[INFO] starting video stream...")
    vs = RealSenseVideo(width=640, height=480).start()

    try:
        while True:
            # Get the video frames from the camera
            color_frame, depth_frame = vs.read() 

            # Extract the dimensions of the depth frame
            (H, W) = depth_frame.get_height(), depth_frame.get_width()

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # and extract the image dimensions
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
            # Downsize frame before feeding it into the object detector
            frame = color_image 
            color_image = np.expand_dims(color_image, axis=0)
            input_tensor = tf.convert_to_tensor(color_image, dtype=tf.float32)
            (detections, predictions_dict, shapes) = detect(input_tensor)

            label_id_offset = 1
            frame = frame.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                detections['detection_boxes'][0].numpy(), 
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.50,
                agnostic_mode=False)

            # Convert the detections and their respective scores from numpy arrays to lists
            DETECTIONS = detections['detection_boxes'][0].numpy().tolist()
            SCORES = detections['detection_scores'][0].numpy().tolist()
            points = get_bb_coordinates(DETECTIONS, SCORES, H, W)

            for point in points:
                # Extract the bounding box coordinates 
                x1, y1, x2, y2 = point
                
                # Find the mid-point coordinates
                midX = (x1+x2)//2
                midY = (y1+y2)//2

                # Find the distance of each point
                dist = filter_distance(depth_frame, midX, midY)

                # Draw a circle at the midpoint for visual validation
                cv2.circle(frame, (midX, midY), radius=5, 
                    color=(0,0,255), thickness=2)  

                # Display the distance of each object from the camera
                text = "Distance: {}cm".format(dist)
                cv2.putText(frame, text, (midX, midY-20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), thickness=2)

                # Determine what command to give to the user
                navigate(frame, depth_frame, dist, x1, x2)
            
            if not checkpoint_detection:
                if checkpoints(depth_frame):
                    command("Stop", frame)
                    time.sleep(0.5)
                else:
                    command("Forward", frame)

            checkpoint_detection = False

            # Display the video frame 
            cv2.namedWindow('RealSense')
            cv2.imshow('RealSense', frame)

            # End the video stream is the letter "Q" is pressed
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                print("[INFO] Ending video stream...")
                break

        # Stop streaming
        vs.stop()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print("Problem: {}".format(e))
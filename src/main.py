'''
    Author: Jordan Madden
    Usage: python main.py --model="ssdmobilenet_v2"
           python main.py --model="efficientdet_d0" 
'''

from playsound import playsound 
import pyrealsense2 as rs
import numpy as np
import argparse
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

def playback(motion_command, cmds):
    #Play audio recording of the given command
    playsound(cmds[motion_command])

def filter_distance(depth_frame, x, y):
    #List to store the consecutive distance values and randomly initialized variable
    distances = []
    positive = np.random.randint(low=30, high=100)

    i = 0
    while(i < 10):
        # Extract the depth value from the camera
        dist = int(depth_frame.get_distance(x, y) * 100)
        
        # Store the last positive value for use in the event that the
        # value returned is 0
        if dist != 0:
            positive = dist
        elif dist == 0:
            dist == positive

        # Add the distances to the list
        distances.append(dist)
        i += 1

    # Convert the list to a numpy array and return it
    distances = np.asarray(distances)
    return int(distances.mean()) 

def get_mid_coordinates_and_dist(depth_frame, detections, scores,
                                H, W, confidence=0.5):
    # Initialize list to store midpoints of each bounding box
    midPoints = []

    for detection, score in zip(detections, scores):
        # Only move forward if score is above the threshold
        if score > confidence:
            # Extract the coordinates of the detections and normalize each detection
            y1, x1, y2, x2 = detection
            y1 = int(H*y1)
            x1 = int(W*x1)
            y2 = int(H*y2)
            x2 = int(W*x2)

            #Display the coordinates and scores of each detection
            print(f"X1: {x1} Y1: {y1} X2: {x2} Y2: {y2}")
            print("Score: {}%".format(score*100))

            # Calculate the midpoint of each box
            midX = int((x1+x2)/2)
            midY = int((y1+y2)/2)
            
            # Calculate the distance of each point
            dist = filter_distance(depth_frame, midX, midY)

            # Add the midpoints to the midpoints list
            midPoints.append([midX, midY, dist])

    return midPoints

if __name__ == "__main__":
    # Construct and parse the command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="type of model to use")
    args = vars(ap.parse_args())

    # Declare the filepaths for text to speech
    COMMAND_PATH = 'text_to_speech/commands'
    FWD = 'forward_command.mp3'
    LEFT = 'left_command.mp3'
    RIGHT = 'right_command.mp3'
    STOP = 'stop_command.mp3'
    forwardPath = os.path.join(COMMAND_PATH, FWD)
    leftPath = os.path.join(COMMAND_PATH, LEFT)
    rightPath = os.path.join(COMMAND_PATH, RIGHT)
    stopPath = os.path.join(COMMAND_PATH, STOP)

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

    # Check distance at these locations 
    center = (W//2, H//2)
    left = (W//2 + 100, H//2)
    right = (W//2 - 100, H//2)
    left2 = (W//2 + 200, H//2)
    right2 = (W//2 - 200, H//2)
    bottom = (W//2, H//2 + 150)
    bottomR = (W//2 - 100, H//2 + 150)
    bottomL = (W//2 + 100, H//2 + 150)
    
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

    # Configure depth and color streams
    print("[INFO] building and configuring the video pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    print("[INFO] starting video stream...")
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue   

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

            # Display the number of detections and their coordinates
            DETECTIONS = detections['detection_boxes'][0].numpy().tolist()
            SCORES = detections['detection_scores'][0].numpy().tolist()
            points = get_mid_coordinates_and_dist(depth_frame, DETECTIONS, SCORES, H, W)

            for point in points:
                # Extract the mid-point dimensions
                midX, midY, dist = point

                # Draw a circle at the midpoint for visual validation
                cv2.circle(frame, (midX, midY), radius=10, 
                    color=(0,0,255), thickness=2)  

                # Display the distance of each object from the camera
                text = "Distance: {}cm".format(dist)
                cv2.putText(frame, text, (midX, midY-20), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 0, 255), thickness=2)

            # Draw vertical lines on the video frame
            cv2.line(frame, (200, 0), (200, 480), (0, 0, 255), 2)
            cv2.line(frame, (400, 0), (400, 480), (0, 0, 255), 2)

            # Stack both images horizontally and display them
            cv2.namedWindow('RealSense')
            cv2.imshow('RealSense', frame)

            #Display the command to be given to the user
            text = "Command: {}".format("Forward")
            cv2.putText(frame, text, (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # End the video stream is the letter "Q" is pressed
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                print("[INFO] Ending video stream...")
                break

        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print("Problem: {}".format(e))



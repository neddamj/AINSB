'''
    Author: Jordan Madden
    Usage: python rpi_test.py
'''

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import argparse
import pathlib
import time
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Construct and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='/home/pi/tensorflow/od_models/my_mobilenet_model')
ap.add_argument('--labels', help='Where the Labelmap is Located',
                    default='/home/pi/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt')
ap.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
args = vars(ap.parse_args())

def visualize_boxes_and_labels(boxes, scores, classes, confidence, H, W):
    count = 0
    
    for i in range(len(scores)):
        if ((scores[i] > confidence) and (scores[i] <= 1.0)):
            count += 1
            
            # Get bounding box coordinates 
            ymin = int(max(1,(boxes[i][0] * H)))
            xmin = int(max(1,(boxes[i][1] * W)))
            ymax = int(min(H,(boxes[i][2] * H)))
            xmax = int(min(W,(boxes[i][3] * W)))
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            
            # Label bounding box
            object_name = category_index[int(classes[i])]['name'] 
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText (frame,'Objects Detected : ' + str(count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)
    
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

            #Display the coordinates and scores of each detection
            print(f"X1: {x1} Y1: {y1} X2: {x2} Y2: {y2}")
            print("Score: {}%".format(score*100))

            # Add the midpoints to the midpoints list
            coordinates.append([x1, y1, x2, y2])

    return coordinates
        
if __name__ == "__main__":
    # Declare relevant constants(filepaths etc)
    PATH_TO_MODEL_DIR = args["model"]
    PATH_TO_LABELS = args["labels"]
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    CONFIDENCE = args["threshold"]

    # Load the model from disk
    print('[INFO] loading model...')
    start_time = time.time()
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print('[INFO] model loaded, took {} seconds'.format(time.time() - start_time))

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    # Configure depth and color streams
    print("[INFO] building and configuring the video pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    print("[INFO] starting video stream...")

    while True:
        # Get the video frames from the camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
                
        # Convert images to numpy arrays and get the frame dimensions
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())
        imH, imW = frame.shape[:2]

        # Convert np array to tensor and add new axis before passing image to detector
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        # Extract the scores, detections and classes
        scores = detections['detection_scores']
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        
        visualize_boxes_and_labels(boxes, scores, classes, CONFIDENCE, imH, imW)
        
        points = get_bb_coordinates(boxes, scores, imH, imW)
        for point in points:
            # Extract the bounding box coordinates 
            x1, y1, x2, y2 = point
            
            # Find the midpoint coordinates
            midX = (x1+x2)//2
            midY = (y1+y2)//2
            
            # Find the distance at each point
            dist = filter_distance(depth_frame, midX, midY)
            
            # Draw a circle at the midpoint for visual validation
            cv2.circle(frame, (midX, midY), radius=5, 
                color=(0,0,255), thickness=2)
            
            # Display the distance of each object from the camera
            text = "Distance: {}cm".format(dist)
            cv2.putText(frame, text, (midX, midY-20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 255), thickness=2)
                        
        
        cv2.imshow('Realsense', frame)

        if cv2.waitKey(1) == ord('q'):
            print("[INFO] ending video stream...")
            break

    # Stop video streaming and destroy thr frame
    pipeline.stop()
    cv2.destroyAllWindows()


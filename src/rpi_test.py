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
                    default='home/pi/od_models/my_mobilenet_model')
ap.add_argument('--labels', help='Where the Labelmap is Located',
                    default='models/research/object_detection/data/mscoco_label_map.pbtxt')
ap.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
args = vars(ap.parse_args())

# Declare relevant constants(filepaths etc)
PATH_TO_MODEL_DIR = 'od_models/my_mobilenet_model'
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
MIN_CONF_THRESH = 0.5

# Load the model and the label map
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

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

    
    # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS
    
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    count = 0
            
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            #increase count
            count += 1
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            # Draw label
            object_name = category_index[int(classes[i])]['name'] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            

    cv2.putText (frame,'Objects Detected : ' + str(count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)
    cv2.imshow('Object Detector', frame)

    if cv2.waitKey(1) == ord('q'):
        print("[INFO] ending video stream...")
        break

pipeline.stop()
cv2.destroyAllWindows()
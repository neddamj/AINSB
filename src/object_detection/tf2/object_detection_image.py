#USAGE: python object_detection_image.py

import os 
import cv2
import numpy as np
from config import PATH_TO_LABELS, path_to_cfg, path_to_ckpt

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    

import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

# Suppress TensorFlow logging (2)
tf.get_logger().setLevel('ERROR')    

print("[INFO] Building model pipeline and detector...")
configs = config_util.get_configs_from_pipeline_file(path_to_cfg(model='ssdmobilenet_v2'))
model_config = configs['model']
detector = model_builder.build(model_config=model_config, is_training=False)

print("[INFO] Restoring model checkpoint...")
PATH_TO_RESTORE = os.path.join(path_to_ckpt(model='ssdmobilenet_v2'), 'ckpt-0')
ckpt = tf.compat.v2.train.Checkpoint(model=detector)
ckpt.restore(PATH_TO_RESTORE).expect_partial()

@tf.function
def detect(img):
    img, shapes = detector.preprocess(img)
    prediction_dict = detector.predict(img, shapes)
    detections = detector.postprocess(prediction_dict, shapes)

    return (detections, prediction_dict, tf.reshape(shapes, [-1]))

def get_mid_coordinates(detections, scores, H, W, confidence=0.5):
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
            midX = (x1+x2)/2
            midY = (y1+y2)/2

            # Add the midpoints to the midpoints list
            midPoints.append([int(midX), int(midY)])

    return midPoints
 
#Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Reading the image
print("[INFO] Reading image...")
frame = cv2.imread("soccer.jpg")
frame2 = frame
frame = np.expand_dims(frame, axis=0)

# Extract image dimensions
(H, W) = frame2.shape[:2]
print(f"\nHeight: {H}\tWidth: {W}")        

input_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
(detections, predictions_dict, shapes) = detect(input_tensor)

label_id_offset = 1
frame2 = frame2.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    frame2,
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
'''print(DETECTIONS[0][0])
print(SCORES[0])'''
midPoints = get_mid_coordinates(DETECTIONS, SCORES, H, W)

for midPoint in midPoints:
    # Extract the mid-point dimensions
    midX, midY = midPoint

    # Draw a circle at the midpoint for visual validation
    cv2.circle(frame2, (midX, midY), radius=10, color=(0,0,255), thickness=2)
    
    # Display the midpoint
    print(f"Mid-X: {midX} Mid-Y: {midY}")      

# Show image
cv2.imshow('Webcam', frame2)    
cv2.waitKey(0)















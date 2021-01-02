#USAGE: python object_detector.py

import os 
import cv2
import numpy as np
from config import PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS

print("Hello")
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    

import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

# Suppress TensorFlow logging (2)
tf.get_logger().setLevel('ERROR')    

print("[INFO] Building model pipeline and detector...")
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detector = model_builder.build(model_config=model_config, is_training=False)

print("[INFO] Restoring model checkpoint...")
PATH_TO_RESTORE = os.path.join(PATH_TO_CKPT, 'ckpt-0')
ckpt = tf.compat.v2.train.Checkpoint(model=detector)
ckpt.restore(PATH_TO_RESTORE).expect_partial()

@tf.function
def detect(img):
    img, shapes = detector.preprocess(img)
    prediction_dict = detector.predict(img, shapes)
    detections = detector.postprocess(prediction_dict, shapes)

    return (detections, prediction_dict, tf.reshape(shapes, [-1]))
    

#Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
cap = cv2.VideoCapture(0)
print("[INFO] Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame2 = frame
    frame = np.expand_dims(frame, axis=0)

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
          min_score_thresh=.6,
          agnostic_mode=False)

    cv2.imshow('Webcam', frame2)    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("[INFO] Ending video stream...")
        break

cap.release()
cv2.destroyAllWindows()















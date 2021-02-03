'''
   Usage: python convert_to_tflite.py
'''
import tensorflow as tf
import argparse

# Construct and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', help='Folder that the saved model is located in',
                    default='exported-models/my_tflite_model/saved_model')
ap.add_argument('--output', help='Folder that the tflite model will be written to',
                    default='exported-models/my_tflite_model')
args = ap.parse_args()

# Convert the model to TF Lite
converter = tf.lite.TFLiteConverter.from_saved_model(args.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

# Save the model to disk
output = args.output + '/model.tflite'
with tf.io.gfile.GFile(output, 'wb') as f:
  f.write(tflite_model)
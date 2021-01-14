'''
    Usage: python compute_depth.py
'''

import pyrealsense2 as rs
import numpy as np
import cv2

def filter_distance(dist):
    # Convert the distance value to a value containing only 2 decimal places
    # and return that value
    whole, frac = str(dist).split(".")
    whole, frac = int(whole), int(int(frac)/10000000000000)
    dist = whole + frac/100
    return dist

# Declare all relevant constants
SCALE = 1.0

# Configure depth and color streams
print("[INFO] building and configuring the video pipeline...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
print("[INFO] starting video stream...")

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # and extract the image dimensions
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        (H, W) = depth_colormap.shape[:2]
        H, W = int(SCALE*H), int(SCALE*W)

        # Find the distance of an arbitraty point in the video frame. The distance 
        dist = depth_frame.get_distance(240, 320)
        print("Distance: {}".format(filter_distance(dist*100)))
        
        # Resize the images to known dimensions
        color_image = cv2.resize(color_image, (W, H))
        color_image = cv2.resize(color_image, (500, 500))
        color_image = cv2.resize(color_image, (W, H))
        depth_colormap = cv2.resize(depth_colormap, (W, H))

        # Stack both images horizontally and display them
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('RealSense')
        cv2.imshow('RealSense', images)

        # Break from loop if the "Q" key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:

    # Stop streaming
    print("[INFO] ending video stream...")
    pipeline.stop()

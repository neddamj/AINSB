'''
    Usage: python compute_depth.py
'''

import pyrealsense2 as rs
import numpy as np
import cv2

def filter_distance(depth_frame, x, y):
    #List to store the consecutive distance values and randomly initialized variable
    distances = []
    positive = np.random.randint(low=30, high=100)

    for i in range(10):
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

    # Convert the list to a numpy array and return it
    distances = np.asarray(distances)
    return int(distances.mean())

# Declare all relevant constants
SCALE_H = 1.0
SCALE_W = 1.0

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
        
        # Extract the dimensions of the depth frame
        (H, W) = depth_frame.get_height(), depth_frame.get_width()
        H, W = int(SCALE_H*H), int(SCALE_W*W)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # and extract the image dimensions
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Find the distance of an arbitraty point in the video frame. The distance 
        dist = filter_distance(depth_frame, W//2, H//2)
        #dist = depth_frame.get_distance(240, 320)
        print("Distance: {}".format(dist))
        
        # Resize the images to known dimensions
        color_image = cv2.resize(color_image, (W, H))
        depth_colormap = cv2.resize(depth_colormap, (W, H))

        # Stack both images horizontally and display them
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('RealSense')
        cv2.imshow('RealSense', images)

        # Break from loop if the "Q" key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] ending video stream...")  
            break

finally:

    # Stop streaming
    pipeline.stop()

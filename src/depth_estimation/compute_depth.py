'''
    Usage: python compute_depth.py
'''

import pyrealsense2 as rs
import numpy as np
import cv2

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
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Find the distance of an arbitraty point in the video frame
        dist = depth.get_distance('''x, y''')

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Resize the images to known dimensions
        cv2.resize(color_image, (224, 224))
        cv2.resize(depth_colormap, (224, 224))

        '''# Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))'''

        # Show images
        cv2.namedWindow('RealSense')
        cv2.imshow('RealSense', color_image)



        # Break from loop if the "Q" key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:

    # Stop streaming
    print("[INFO] ending video stream...")
    pipeline.stop()

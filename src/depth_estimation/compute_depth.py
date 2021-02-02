'''
    Author: Jordan Madden
    Usage: python compute_depth.py --device="rpi"
           python compute_depth.py --device="win" 
'''
from imutils.video import FPS
import numpy as np
import argparse
import cv2

# Construct and parse the command arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--device", required=True,
                 help="Will the code be run on raspberry pi or windows machine?")
args = vars(ap.parse_args())

if args["device"] == "rpi":
    import pyrealsense2.pyrealsense2 as rs
elif args["device"] == "win":
    import pyrealsense2 as rs

def filter_distance(depth_frame, x, y):
    #List to store the consecutive distance values and randomly initialized variable
    distances = []
    positive = np.random.randint(low=30, high=100)

    i = 0
    while(i < 50):
        # Extract the depth value from the camera
        dist = int(depth_frame.get_distance(x, y) * 100)
        
        # Store the last positive value for use in the event that the
        # value returned is 0
        if dist != 0:
            positive = dist
        elif dist == 0:
            positive = positive

        # Add the distances to the list
        distances.append(positive)
        i += 1

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
fps = FPS().start()
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

        # Update the FPS counter
        fps.update()

        # Break from loop if the "Q" key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] ending video stream...")  
            fps.stop()
            break

finally:
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # Stop streaming
    pipeline.stop()

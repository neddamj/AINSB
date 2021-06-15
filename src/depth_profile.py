'''
    Author: Jordan Madden
    Usage: python depth_profile.py
'''

from realsense import RealSense, filter_distance
import matplotlib.pyplot as plt
from imutils.video import FPS
import numpy as np
import time
import cv2

def get_depth_profile(depth_frame, profile_width, X, Y):
    # Ensure the coordiantes fall within the frame
    if (profile_width >= X):
        profile_width = X
    elif (profile_width >= 640-X):
        profile_width = 640-X
    
    # Array to store the depth profile
    depth_profile = []

    # Get distance of each pixel along the profile width and add it to 
    # the depth profile list
    for i in range(profile_width):
        depth_profile.append(filter_distance(depth_frame, X+i, Y))

    return np.array(depth_profile)

if __name__ == "__main__":
    video = RealSense(width=640, height=480).start()
    fps = FPS().start()
    
    while True:
        color_frame, depth_frame = video.read()
        
        # Convert images to numpy arrays and get the frame dimensions
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw line on screen to visulalize 300px
        cv2.line(frame, (100, 240), (600, 240), (0, 160, 200), 2)
        #depth = depth_image[100:500,240:240].astype(float)*depth_scale
        
        # Get the depth profile as a array
        pts = get_depth_profile(depth_frame, 500, 350, 240)
        
        # Plot the depth profile on a graph
        print(pts)
        plt.plot(pts)
        plt.xlabel("Position/pixel")
        plt.ylabel("Distance/cm")
        plt.title("Depth Profile (Depth v Position)")
        plt.show()
        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object Detector', frame)
        fps.update()
        
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            fps.stop()
            break
        
    # Show the elapsed time and the respective fps
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approximate fps: {:.2f}".format(fps.fps()))
    
    # Stop the video stream
    cv2.destroyAllWindows()
    video.stop()    
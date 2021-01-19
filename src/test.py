'''
    Author: Jordan Madden
    Usage: python test.py  
'''

import numpy as np 
import cv2

if __name__ == "__main__":
    # Declare relevant constants
    H = 480
    W = 600

    # Create the video capture object
    cap = cv2.VideoCapture(0)
    print("[INFO] starting video feed...")

    while True:
        # Read each frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to the size that I will be working with
        frame = cv2.resize(frame, (W, H))

        # Draw vertical lines on the video frame
        cv2.line(frame, (200, 0), (200, 480), (0, 0, 255), 2)
        cv2.line(frame, (400, 0), (400, 480), (0, 0, 255), 2)

        # Show the image 
        cv2.imshow("Webcam", frame)

        # Stop video if the letter "Q" is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] ending video stream...")
            break
        

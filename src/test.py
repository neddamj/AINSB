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

    # Check distance at these locations 
    center = (W//2, H//2)
    left = (W//2 + 100, H//2)
    right = (W//2 - 100, H//2)
    left2 = (W//2 + 200, H//2)
    right2 = (W//2 - 200, H//2)
    bottom = (W//2, H//2 + 150)
    bottomR = (W//2 - 100, H//2 + 150)
    bottomL = (W//2 + 100, H//2 + 150)

    # Create the video capture object
    print("[INFO] starting video feed...")
    cap = cv2.VideoCapture(0)
    
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

        # Draw circles at various points around the frame
        cv2.circle(frame, center, 10, (0, 0, 255), 1)
        cv2.circle(frame, right, 10, (0, 0, 255), 1)
        cv2.circle(frame, left, 10, (0, 0, 255), 1)
        cv2.circle(frame, right2, 10, (0, 0, 255), 1)
        cv2.circle(frame, left2, 10, (0, 0, 255), 1)
        cv2.circle(frame, bottom, 10, (0, 0, 255), 1)
        cv2.circle(frame, bottomR, 10, (0, 0, 255), 1)
        cv2.circle(frame, bottomL, 10, (0, 0, 255), 1)

        # Show the image 
        cv2.imshow("Webcam", frame)

        # Stop video if the letter "Q" is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] ending video stream...")
            break
        

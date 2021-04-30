'''
    Author: Jordan Madden
    Desccription: Allows the efficient use of the Intel Realsense D415 camera by reading the
                  data from the camera on its own thread
            
'''

import pyrealsense2.pyrealsense2 as rs
from threading import Thread

class RealSense:
    def __init__(self, width=640, height=480):
        # Frame dimensions of camera
        self.width = width
        self.height = height

        # Build and enable the depth and color frames
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
        
        # Read the first frame from the stream
        self.frame = self.pipeline.wait_for_frames()
        self.depth_frame = self.frame.get_depth_frame()
        self.color_frame = self.frame.get_color_frame()
        
        # Variable to check if thread should be stopped
        self.stopped = False

    def start(self):
        # Start the thread to read the frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            # Stop streaming in indicator is set
            if self.stopped:
                return

            # Otherwise read the next frame in the stream
            self.frame = self.pipeline.wait_for_frames()
            self.depth_frame = self.frame.get_depth_frame()
            self.color_frame = self.frame.get_color_frame()
            
            if not self.depth_frame or not self.color_frame:
                return

    def read(self):
        # Return the most recent color and depth frame
        return self.color_frame, self.depth_frame
    
    def filter_depth(self, depth):
        # Apply post processing filters to depth image
        spat_filter = rs.spatial_filter()
        temp_filter = rs.temporal_filter()
        depth = spat_filter.process(depth)
        depth = temp_filter.process(depth)
        
        return depth.as_depth_frame()

    def stop(self)        :
        # Stop the video stream
        self.stopped = True
        self.pipeline.stop()
from util import *

import numpy as np
import pyrealsense2 as rs

class RealSenseCamera:

    def __init__(self) -> None:

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        log(f"Device in use: {self.device}")

        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                break

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

    @staticmethod
    def frame2Pixles(frame):
        return np.asanyarray(frame.get_data()) if frame else None

    def getFrames(self):

        frames = self.pipeline.wait_for_frames()  

        colorPixels = self.frame2Pixles(frames.get_color_frame())
        depthPixels = self.frame2Pixles(frames.get_depth_frame()) 
        
        if colorPixels is None:
            warn(f"Failed to capture realsense color frame")

        if depthPixels is None:
            warn(f"Failed to capture realsense depth frame")

        return colorPixels, depthPixels   
    




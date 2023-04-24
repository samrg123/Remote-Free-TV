from util import *
from frame import Frame

import time
import threading
import numpy as np

class DepthCamera:

    class FrameGrabber(threading.Thread):

        def __init__(self, depthCamera:'DepthCamera') -> None:
            super().__init__(daemon=True)
            
            self.stop = False
            self.started = False
            self.depthCamera = depthCamera

        def start(self) -> None:
            self.started = True
            return super().start()

        def run(self) -> None:
            frameId = 0
            lastFrameTime = time.time()
            while not self.stop:
            
                frameTime = time.time()
                deltaTime = frameTime - lastFrameTime
                
                colorPixels, depthPixels = self.depthCamera.getPixels()
              
                frame = Frame(
                    id = frameId,
                    time = frameTime,
                    frameRate = 1/deltaTime,
                    colorPixels = colorPixels,
                    depthPixels = depthPixels
                )

                self.depthCamera.lock.acquire()
                self.depthCamera.frame = frame
                self.depthCamera.lock.release()

                frameId+= 1
                lastFrameTime = frameTime

    def __init__(self) -> None:
        self.frame = Frame()
        self.lock = threading.Lock()
        self.frameGrabber = self.FrameGrabber(self)

    def start(self) -> None:
        self.frameGrabber.start()

    def getPixels(self) -> tuple[np.array, np.array]:
        warn(f"getPixels not implemented for: {self}")
        return None, None

    def __del__(self) -> None:
        if self.frameGrabber.started:
            self.frameGrabber.stop = True
            self.frameGrabber.join(timeout=3)

    def getFrame(self) -> Frame:
        self.lock.acquire()
        frame = self.frame
        self.lock.release()

        return frame

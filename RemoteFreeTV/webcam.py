from util import *
from cvUtil import *

from depthCamera import *

class DepthWebcam(DepthCamera):
        
    def __init__(
        self, 
        port:int = 0, 
        width:int = 1280, 
        height:int = 720, 
        fps:int = 30,
        bufferSize:int = 2
    ) -> None:
        super().__init__()
        
        self.port = port
        self.camera = cv.VideoCapture(port)

        # Note: Needed to get camera decoding on linux
        self.camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))

        self.width = width
        self.height = height

        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, height) 

        self.fps = fps
        self.camera.set(cv.CAP_PROP_FPS, fps)

        self.bufferSize = bufferSize
        self.camera.set(cv.CAP_PROP_BUFFERSIZE, bufferSize)
        
        super().start()

    def getPixels(self):
        validFrame, colorPixels = self.camera.read()

        if not validFrame:
            warn(f"Failed to capture video frame on port: {self.port}")

        return colorPixels, None
import argparse

import math
import time
import numpy as np

from util import *
from cvUtil import *
from mpUtil import *

from realsense import *
from webcam import *

from handAnnotations import *

from rokuECP import RokuECP
from gestures import *

class GestureRecognizer:
    def __init__(
        self,
        depthCamera:DepthCamera,
        modelPath:str,
        rokuUrl=RokuECP.defaultUrl,
        windowName: str = None,
        headless=False,
        asyncUpdate=False
    ) -> None:
        
        self.headless = headless
        self.depthCamera:DepthWebcam|RealSenseCamera = depthCamera
        self.rokuEcp = RokuECP(rokuUrl)

        if asyncUpdate:
            self.update = self.updateAsync
            mpRunningMode = MPRunningMode.LIVE_STREAM
            mpCallback = self.mpGestureRecognizerCallback
        else:
            self.update = self.updateSync    
            mpRunningMode = MPRunningMode.VIDEO
            mpCallback = None

        log(f"Loading Model: '{modelPath}'")
        self.GestureRecognizerOptions = MPGestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=modelPath
            ),
            running_mode=mpRunningMode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            # TODO: Figure out how to get canned gestures working
            # canned_gesture_classifier_options = MakeMPClassifierOptions(
            #     score_threshold = 0, # return all scores
            #     # category_allowlist = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"],
            #     category_allowlist = ["None", "Closed_Fist", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory"],
            #     category_denylist = []
            # ),
            # custom_gesture_classifier_options = MakeMPClassifierOptions(),
            
            # Note: Ignored for non LIVE_STREAM running modes
            result_callback=mpCallback,
        )

        self.mpGestureRecognizer = (
            MPGestureRecognizer.create_from_options(
                self.GestureRecognizerOptions
            )
        )

        if not headless:
            self.windowName = (
                windowName
                if windowName is not None
                else f"GestureRecognizer: {type(depthCamera).__name__}"
            )
            cv.namedWindow(self.windowName, cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO )

        # TODO: Clean this up!
        self.lock = threading.Lock()
        self.processingFrames = {}

        self.lastUpdateTime = time.time()
        self.lastMpImageFrameTimeMs = 0

        self.renderImages: list[NamedImage] = []

    def __del__(self):
        if self.mpGestureRecognizer:
            self.mpGestureRecognizer.close()

    def __bool__(self):
        return (
            self.headless
            or cv.getWindowProperty(self.windowName, cv.WND_PROP_VISIBLE) == 1
        )
    
    def detectGestures(self, handAnnotations) -> list[Gesture]:
        gestures = []
        for gesture in Gestures:
            if gesture.isDetected(handAnnotations):
                gestures.append(gesture)

        return gestures    

    def processHandAnnotations(self, handAnnotations:HandAnnotations, frame:Frame, updateRate:float):
        
        detectedGestures = self.detectGestures(handAnnotations)
        for gesture in detectedGestures:
            self.rokuEcp.sendCommand(gesture.rokuKey, gesture.rokuCommand)

        if self.headless:
            print(f"FPS: {updateRate:.2f} / {frame.frameRate:.2f}", end="\r")

        else:
            processedPixels = frame.colorPixels
            handAnnotations.draw(processedPixels)

            # TODO: append depth image if we use it
            self.renderImages = [
                NamedImage(f"Processed: {updateRate:.2f} / {frame.frameRate:.2f} FPS", processedPixels)
            ]

    def mpGestureRecognizerCallback(self, mpResult:MPGestureRecognizerResult, mpImage:mp.Image, timestampMs:int):
        updateTime = time.time()

        self.lock.acquire()
        frame = self.processingFrames.pop(timestampMs)
        lastUpdateTime = self.lastUpdateTime
        self.lock.release()

        handAnnotations = HandAnnotations(frameId=frame.id, frameTime=frame.time)
        handAnnotations.addMPGestureRecognizerResult(mpResult)
        
        updateRate = 1 / (updateTime - lastUpdateTime)
        self.processHandAnnotations(handAnnotations, frame, updateRate)

        self.lock.acquire()
        if updateTime > self.lastUpdateTime:
            self.lastUpdateTime = updateTime
        self.lock.release()

    def getNewMpImage(self, frame:Frame) -> MPImage:
        
        if frame.colorPixels is None:
            return None

        # Note: mediapipe will throw exception if we try to process a frame with the same int timestamp 
        frameTimeMs = frame.timeMs()
        if frameTimeMs <= self.lastMpImageFrameTimeMs:
            return None
        
        # Note: Media pipe uses RGB for CNN models
        # TODO: Do we need to convert cv2 BGR frame into RGB? does realsense give use an RGB frame... test this out
        #       and check whether or not switching green and blue pixels improves hand gesture recognition
        mpImage = MPImage(image_format=MPImageFormat.SRGB, data=frame.colorPixels)

        self.lastMpImageFrameTimeMs = frameTimeMs
        return mpImage

    def updateAsync(self) -> None:
        frame = self.depthCamera.getFrame()

        mpImage = self.getNewMpImage(frame)
        if mpImage is None:
            return

        frameTimeMs = frame.timeMs()

        self.lock.acquire()

        if len(self.processingFrames) < 5:        
            self.processingFrames[frameTimeMs] = frame
            self.mpGestureRecognizer.recognize_async(
                mpImage, frameTimeMs
            )
     
            # print(len(self.processingFrames))

        self.lock.release()

    def updateSync(self) -> None:
        frame = self.depthCamera.getFrame()     
        
        mpImage = self.getNewMpImage(frame)
        if mpImage is None:
            return
       
        updateTime = time.time()
        frameTimeMs = frame.timeMs()

        handAnnotations = HandAnnotations(frameId=frame.id, frameTime=frame.time)
        mpResult = self.mpGestureRecognizer.recognize_for_video(
            mpImage, frameTimeMs
        )
        handAnnotations.addMPGestureRecognizerResult(mpResult)

        updateRate = 1 / (updateTime - self.lastUpdateTime)
        self.processHandAnnotations(handAnnotations, frame, updateRate)

        self.lastUpdateTime = updateTime
        
    def draw(self) -> None:
        
        # Note: We still call cv.waitKey if headless so camera has time to breath
        if self.headless:
            cv.waitKey(1)
            return

        # Create a blank window image buffer
        _, _, windowWidth, windowHeight = cv.getWindowImageRect(
            self.windowName
        )
        windowImage = np.zeros((windowHeight, windowWidth, 3), dtype=np.uint8)

        # Compute layout of image grid
        numImages = len(self.renderImages)
        numXImages = max(1, int(math.ceil(np.sqrt(numImages))))
        numYImages = max(1, int(math.ceil(numImages / numXImages)))

        maxImageWidth = int(windowWidth / numXImages)
        maxImageHeight = int(windowHeight / numYImages)

        maxImageAspect = maxImageWidth / maxImageHeight

        # Blit images to window image buffer
        for i in range(0, numImages):
            row = i // numXImages
            col = i - row * numXImages

            y = row * maxImageHeight
            x = col * maxImageWidth

            image = self.renderImages[i].getImage()
            imageAspect = image.shape[1] / image.shape[0]

            if imageAspect >= maxImageAspect:
                # Fit to width of image
                imageWidth = maxImageWidth
                imageHeight = int(imageWidth / imageAspect + 0.5)
                y += (maxImageHeight - imageHeight) // 2

            else:
                # Fit to height of image
                imageHeight = maxImageHeight
                imageWidth = int(imageAspect * imageHeight + 0.5)
                x += (maxImageWidth - imageWidth) // 2

            resizedImage = cv.resize(
                image, (imageWidth, imageHeight), interpolation=cv.INTER_LANCZOS4
            )

            # Note: We add empty color dimension to the resized image if its monochromatic so numpy
            #       can broadcast it to 3 color channel destination
            if len(resizedImage.shape) == 2:
                resizedImage = resizedImage[..., np.newaxis]

            windowImage[y : y + imageHeight, x : x + imageWidth] = resizedImage

        # Display window image buffer
        # Note: We need to pause via waitKey to allow opencv to display frame
        cv.imshow(self.windowName, windowImage)
        cv.waitKey(1)


def main():

    verboseLevel = getVerboseLevel()

    argParser = argparse.ArgumentParser(
        prog="gestureRecognition",
        description="Gesture classifier for Remote Free TV",
    )

    argParser.add_argument(
        "-l",
        "--list",
        action="store_true",
        default=False,
        required=False,
        help="List available port numbers",
    )
    argParser.add_argument(
        "-r",
        "--realsense",
        action="store_true",
        required=False,
        help="Uses Intel RealSense depth camera. If this flag isn't specified webcam is used instead.",
    )
    argParser.add_argument(
        "-w",
        "--webcam",
        metavar="int:port",
        action="store",
        default=None,
        required=False,
        help="Uses webcam at `port`",
    )
    argParser.add_argument(
        "--url",
        required=False,
        action="store",
        metavar="URL",
        default=RokuECP.defaultUrl,
        help=f"Sets the url for connecting to the Roku. Default: '{RokuECP.defaultUrl}'",
    )
    argParser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        default=False,
        required=False,
        help="Disables video output to reduce latency",
    )

    argParser.add_argument(
        "-v",
        "--verbose",
        action="store",
        metavar="int:level",
        default=verboseLevel.value,
        required=False,
        help=f"Sets to verbose logging level. LogLevels: {[str(x) for x in sorted(LogLevel.logLevelList())]}. Default: '{verboseLevel}'",
    )

    defaultModel = "models/hagrid_120k/model/gesture_recognizer.task"
    argParser.add_argument(
        "-m",
        "--model",
        action="store",
        metavar="str:path",
        default=defaultModel,
        required=False,
        help=f"Sets the GestureRecognizerModelPath. Default: '{defaultModel}'",
    )

    args = argParser.parse_args()

    verboseLevel = setVerboseLevel(LogLevel.fromValue(int(args.verbose)))
    print(f"Set verbose level to: '{verboseLevel}'")

    if bool(args.list):
        listVideoPorts()
        return

    if not args.realsense and args.webcam is None:
        panic(f"Must specify either '--webcam' or '--realsense'")

    if args.realsense and args.webcam is not None:
        panic(f"Cannot specify both '--webcam' and '--realsense'")

    if args.realsense:
        camera = RealSenseCamera()
    else:
        # TODO Pass width and height as args!
        camera = DepthWebcam(port=int(args.webcam), width=1920, height=1080)

    gestureRecognizer = GestureRecognizer(
        depthCamera=camera, modelPath=args.model, rokuUrl=args.url, headless=args.headless,

        # TODO: Make sure gestures are thread safe and then try experimenting with this
        # Also experiment with number of async frames (can pass them in as args like: '--async 3')
        asyncUpdate=False
    )

    while gestureRecognizer:
        gestureRecognizer.update()

        # TODO: place drawing on separate thread
        #       so it doesn't bog down queuing up async frames 
        gestureRecognizer.draw()


if __name__ == "__main__":
    main()

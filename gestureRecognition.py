import argparse

import math
import time
import cv2 as cv
import numpy as np

from util import *
from cvUtil import *

from realsense import RealSenseCamera
from webcam import DepthWebcam

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.processors import ClassifierOptions as MPClassifierOptions

from gestures import Gesture
from rokuECP import RokuECP

# TODO: Clean this up ... right now these are hacks to get around nasty wrapper types. Consider MediaPipe util?

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def MakeMPClassifierOptions(
    display_names_locale = "en",
    max_results = -1,
    score_threshold = 0,
    category_allowlist = [],
    category_denylist = [],
) -> MPClassifierOptions:
    
    options = MPClassifierOptions()
    
    options.display_names_locale = display_names_locale
    options.max_results = max_results
    options.score_threshold = score_threshold
    options.category_allowlist = category_allowlist
    options.category_denylist = category_denylist

    return options

def MakeNormalizedLandmarkList(landmarks) -> landmark_pb2.NormalizedLandmarkList:
    landmarkProto = landmark_pb2.NormalizedLandmarkList()
        
    landmarkProto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
    ])

    return landmarkProto

class GestureRecognizer:

    DEBOUNCE_LENGTH:int = 15

    def __init__(self, depthCamera, rokuUrl = RokuECP.defaultUrl, windowName:str = None) -> None:

        self.depthCamera = depthCamera
        self.rokuEcp = RokuECP(rokuUrl)

        self.GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions(

            base_options = mp.tasks.BaseOptions(model_asset_path='model/gesture_recognizer.task'),
            
            running_mode = mp.tasks.vision.RunningMode.VIDEO,
            num_hands    = 2,
            min_hand_detection_confidence = .5,
            min_hand_presence_confidence = .5,
            min_tracking_confidence = .5,

            # TODO: Figure out how to get canned gestures working
            # canned_gesture_classifier_options = MakeMPClassifierOptions(
            #     score_threshold = 0, # return all scores
            #     # category_allowlist = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"],
            #     category_allowlist = ["None", "Closed_Fist", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory"],
            #     category_denylist = []
            # ),

            # custom_gesture_classifier_options = MakeMPClassifierOptions(),

            # Note: Needed for LIVE_STREAM running_mode
            result_callback = None
        )

        self.gestureRecognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.GestureRecognizerOptions)


        self.windowName = windowName if windowName is not None else f"Classifier: {type(depthCamera).__name__}"
        cv.namedWindow(self.windowName, cv.WINDOW_NORMAL|cv.WINDOW_KEEPRATIO)

        self.frameId = 0
        self.frameTime = time.time()

        self.renderImages:list[NamedImage] = []

        self.history = np.empty((self.DEBOUNCE_LENGTH), dtype=object)
        self.lastGesture = Gesture.Nothing
        self.lastGestureId = 0


    def __del__(self):

        if self.gestureRecognizer:
            self.gestureRecognizer.close()

    def __bool__(self):
        return cv.getWindowProperty(self.windowName, cv.WND_PROP_VISIBLE) == 1

    def detectHands(self, framePixels, timestampMS):

        # Note: Media pipe uses RGB for CNN models
        # Note: we use non-writeable flag to pass by reference
        framePixels.flags.writeable = False

        # mpImage = cv.cvtColor(framePixels, cv.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=framePixels)

        # with mp.tasks.vision.GestureRecognizer.create_from_options(self.GestureRecognizerOptions) as r:
            # results = r.recognize_for_video(mpImage, int(timestampMS))

        results = self.gestureRecognizer.recognize_for_video(mpImage, int(timestampMS))

        framePixels.flags.writeable = True

        if not results:
            return 

        hand_annotations = list(zip(results.hand_landmarks, results.handedness, results.gestures))

        for (landmarks, handedness, gesture) in hand_annotations:

            # TODO: Draw the box and label around the image showing what choice the classifier should use!

            print(f'Handedness: {handedness}')
            print(f'Gesture: {gesture}')

            mp_drawing.draw_landmarks(
                framePixels,
                MakeNormalizedLandmarkList(landmarks),
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        return hand_annotations


    def detectGesture(self, handHistory):

        if handHistory[0] is not None and handHistory[-1] is not None:

            if handHistory[0] - handHistory[-1] > 0.2:
                return Gesture.RightToLeft

            elif handHistory[-1] - handHistory[0] > 0.2:
                return Gesture.LeftToRight

        return Gesture.Nothing

    def update(self):
        
        colorFrame, depthFrame = self.depthCamera.getFrames()
        
        if colorFrame is None:
            return

        # clear out last frame
        self.renderImages.clear()

        frameTime = time.time()
        frameRate = 1. / (frameTime - self.frameTime)

        # TODO: Add depth filtering to processedPixels and see how it improves hand / gesture detection
        processedPixels = colorFrame.copy()

        self.history = np.roll(self.history, 1)
        hand_annotations = self.detectHands(processedPixels, 1000*frameTime)  
        
        if hand_annotations:   
            # From user's perspective, x,y at 0,0 is in top right, 1,1 is bottom left
            for (landmarks, handedness, gesture) in hand_annotations:
                self.history[0] = landmarks[0].x
                break
        else:
            self.history[0] = None

        # TODO: make use of new hand_annotations 'gesture' value for static gesture detection
        gesture = self.detectGesture(self.history)
        if gesture != self.lastGesture:
            self.lastGesture = gesture
            self.lastGestureId = self.frameId
            self.rokuEcp.sendGesture(gesture)
            
        self.renderImages.append(NamedImage(f"Raw: {frameRate:.2f} FPS", colorFrame))
        self.renderImages.append(NamedImage(f"Processed ({gesture})", processedPixels))
        
        self.frameId+= 1 
        self.frameTime = frameTime


    def draw(self):
        
        # Create a blank window image buffer
        _, _, windowWidth, windowHeight = windowRect = cv.getWindowImageRect(self.windowName)
        windowImage = np.zeros((windowHeight, windowWidth, 3), dtype=np.uint8)

        # Compute layout of image grid 
        numImages = len(self.renderImages)
        numXImages = max(1, int(math.ceil(np.sqrt(numImages))))
        numYImages = max(1, int(math.ceil(numImages/numXImages)))

        maxImageWidth  = int(windowWidth/numXImages)
        maxImageHeight = int(windowHeight/numYImages)

        maxImageAspect = maxImageWidth/maxImageHeight

        # Blit images to window image buffer
        for i in range(0, numImages):
                
            row = i//numXImages
            col = i - row*numXImages

            y = row*maxImageHeight
            x = col*maxImageWidth

            image = self.renderImages[i].getImage()
            imageAspect = image.shape[1] / image.shape[0]
    
            if imageAspect >= maxImageAspect:
    
                # Fit to width of image
                imageWidth = maxImageWidth
                imageHeight = int(imageWidth/imageAspect + .5)
                y+= (maxImageHeight - imageHeight)//2

            else:
                # Fit to height of image
                imageHeight = maxImageHeight
                imageWidth = int(imageAspect*imageHeight + .5)
                x+= (maxImageWidth - imageWidth)//2

            resizedImage = cv.resize(image, (imageWidth, imageHeight), interpolation=cv.INTER_LANCZOS4)

            # Note: We add empty color dimension to the resized image if its monochromatic so numpy
            #       can broadcast it to 3 color channel destination   
            if(len(resizedImage.shape) == 2):
                resizedImage = resizedImage[..., np.newaxis]

            windowImage[y:y+imageHeight, x:x+imageWidth] = resizedImage 

        # Display window image buffer
        # Note: We need to pause via waitKey to allow opencv to display frame
        cv.imshow(self.windowName, windowImage)
        cv.waitKey(1)
       

def main():

    argParser = argparse.ArgumentParser(
        prog = "gestureRecognition",
        description ="Gesture classifier for Remote Free TV",
    )

    argParser.add_argument("-l", "--list", action="store_true", default=False, required=False, help="List available port numbers")
    argParser.add_argument("-r", "--realsense", action="store_true", required=False, help="Uses Intel RealSense depth camera. If this flag isn't specified webcam is used instead.")
    argParser.add_argument("-w", "--webcam", metavar="int:port", action="store", default=None, required=False, help="Uses webcam at `port`")
    argParser.add_argument("--url",     required=False, action="store", metavar="URL", default=RokuECP.defaultUrl, help=f"Sets the url for connecting to the Roku. Default: '{RokuECP.defaultUrl}'")

    args = argParser.parse_args()

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
        camera = DepthWebcam(port = int(args.webcam))

    classifier = GestureRecognizer(depthCamera=camera, rokuUrl=args.url, windowName="GestureRecognizer")

    while classifier:
        classifier.update()
        classifier.draw()


if __name__ == "__main__":
    main()
    
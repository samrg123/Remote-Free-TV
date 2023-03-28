import platform
import argparse
import math
import time
import cv2 as cv
import numpy as np

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.processors import ClassifierOptions as MPClassifierOptions

# TODO: Clean this up ... right now these are hacks to get around nasty wrapper types

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


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class LogLevel:
    Debug = 2
    Warn  = 1
    Error = 0

verboseLevel:int = LogLevel.Debug

def log(msg, prefix:str = "MSG", logLevel:int = LogLevel.Debug):
    if(verboseLevel >= logLevel):
        print(f"{prefix} - {msg}")

def warn(msg):
    log(msg, prefix="WARN", logLevel=LogLevel.Warn)

def error(msg):
    log(msg, prefix="ERROR", logLevel=LogLevel.Error)


# NOTE: Colors are in BGR to align with opencv
class Color:
    
    white = (255, 255, 255)
    black = (  0,   0,   0)
    red   = (  0,   0, 255)
    green = (  0, 255,   0)
    blue  = (255,   0,   0)

class NamedImage:

    def __init__(self, name:str, pixels:cv.Mat, nameOrigin:tuple[int, int] = None) -> None:
        self.name = name
        self.pixels = pixels

        self.font = cv.FONT_HERSHEY_DUPLEX
        self.fontColor = Color.white
        self.fontSize = .75
        self.fontThickness = 1

        self.fontShadowColor = Color.black
        self.fontShadowSize = 2

        self.defaultFontPadding = [10, 10]

        self.setNameOrigin(nameOrigin)

    def getTextSize(self, text:str):
        textWidth, textHeight  = cv.getTextSize(text, self.font, self.fontSize, self.getShadowThickness())[0]
        return (textWidth, textHeight)

    def setNameOrigin(self, nameOrigin:tuple[int, int]):

        if nameOrigin is None:

            _, nameHeight  = self.getTextSize(self.name)
            self.nameOrigin = (self.defaultFontPadding[0], nameHeight + self.defaultFontPadding[1])
        
        else:
            self.nameOrigin = nameOrigin 

    def getShadowThickness(self):
        return self.fontThickness + self.fontShadowSize

    def drawText(self, text:str, origin:tuple[int, int]):

        # Note: opencv requires integer position for text
        intOrigin = (int(origin[0]), int(origin[1]))

        shadowImage = cv.putText(self.pixels, text, intOrigin, self.font, self.fontSize, self.fontShadowColor, self.getShadowThickness(), cv.LINE_AA)
        return cv.putText(shadowImage, text, intOrigin, self.font, self.fontSize, self.fontColor, self.fontThickness, cv.LINE_AA)

    def getImage(self):
        return self.drawText(self.name, self.nameOrigin)

class Classifier:

    def __init__(self, cameraPort:int = 0, windowName:str = None) -> None:


        self.camera_port = cameraPort
        self.camera = cv.VideoCapture(cameraPort)

        # Note: Needed to get camera decoding on linux
        self.camera.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        self.cameraHeight = 720
        self.cameraWidth  = 1280

        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.cameraHeight)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, self.cameraWidth)

        self.cameraFPS = 30
        self.camera.set(cv.CAP_PROP_FPS, self.cameraFPS)

        # HACK to get around lack of OS support
        # TODO: Windows doesn't support loading models (smh)
        if platform.system() == "Windows":

            self.detectHands = self.detectHandsWindows

        else:

            self.detectHands = self.detectHandsLinux

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

        self.windowName = windowName if windowName is not None else f"Classifier: {cameraPort}"
        cv.namedWindow(self.windowName, cv.WINDOW_NORMAL|cv.WINDOW_KEEPRATIO)

        self.frameId = 0
        self.frameTime = time.time()

        self.renderImages:list[NamedImage] = []

    def __del__(self):
        
        if self.gestureRecognizer:
            self.gestureRecognizer.close()

    def __bool__(self):
        return cv.getWindowProperty(self.windowName, cv.WND_PROP_VISIBLE) == 1

    def detectHandsLinux(self, framePixels, timestampMS):

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

        for (landmarks, handedness, gesture) in zip(results.hand_landmarks, results.handedness, results.gestures):

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

    def detectHandsWindows(self, framePixels, timestampMS):

        with mp_hands.Hands(
            static_image_mode=False, # False = images are part of video stream
            max_num_hands=2,            
            model_complexity=0,      # 1 for more complex gestures (heavy processing)
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,

        ) as hands:


            # Note: Media pipe uses RGB for CNN models
            # Note: we use non-writeable flag to pass by reference
            framePixels.flags.writeable = False

            mpImage = cv.cvtColor(framePixels, cv.COLOR_BGR2RGB)
            results = hands.process(mpImage)

            framePixels.flags.writeable = True

            # Draw the hand annotations on the image.
            if not results.multi_hand_landmarks or not results.multi_handedness:
                return

            for (landmarks, handedness) in zip(results.multi_hand_landmarks, results.multi_handedness):

                print(f'Handedness: {handedness}')

                mp_drawing.draw_landmarks(
                    framePixels,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )


    def update(self):
        
        # clear out last frame
        self.renderImages.clear()

        validFrame, framePixels = self.camera.read()

        if not validFrame:
            warn(f"Failed to capture frame: {self.frameId}")
            return

        frameTime = time.time()
        frameRate = 1. / (frameTime - self.frameTime)


        # TODO: Add depth filtering to processedPixels and see how it improves hand / gesture detection
        processedPixels = framePixels.copy()

        self.detectHands(processedPixels, 1000*frameTime)

        self.renderImages.append(NamedImage(f"Raw: {frameRate:.2f} FPS", framePixels))
        self.renderImages.append(NamedImage(f"Processed", processedPixels))
        
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
        prog = "gestures",
        description ="Gesture classifier for Remote Free TV",
    )

    argParser.add_argument("-p", "--port", metavar="n", action="store", default=0, required=False, help="Camera port number")

    args = argParser.parse_args()

    classifier = Classifier(cameraPort=int(args.port), windowName="Classifier")

    while classifier:
        classifier.update()
        classifier.draw()


if __name__ == "__main__":
    main()
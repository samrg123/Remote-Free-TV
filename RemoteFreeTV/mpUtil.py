
import mediapipe as mp

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark as MPNormalizedLandmark
from mediapipe.tasks.python.components.containers.category import Category as MPCategory

from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer as MPGestureRecognizer
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerOptions as MPGestureRecognizerOptions
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult as MPGestureRecognizerResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as MPRunningMode

from mediapipe.python import Image as MPImage
from mediapipe.python import ImageFormat as MPImageFormat

from mediapipe.python.solutions import drawing_utils as mpDrawing 
from mediapipe.python.solutions import drawing_styles as mpDrawingStyles 
from mediapipe.python.solutions import hands as mpHands 
from mediapipe.python.solutions.hands import HandLandmark as MPHandLandmark 

from mediapipe.framework.formats import landmark_pb2

from mediapipe.tasks.python.components.processors import (
    ClassifierOptions as MPClassifierOptions
)

def MakeMPClassifierOptions(
    display_names_locale="en",
    max_results=-1,
    score_threshold=0,
    category_allowlist=[],
    category_denylist=[],
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

    landmarkProto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in landmarks
        ]
    )

    return landmarkProto

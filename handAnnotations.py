
from dataclasses import dataclass, field

from mpUtil import *
from cvUtil import *
import numpy as np

@dataclass
class HandFeatures:
    landmarks: list[MPNormalizedLandmark] = field(default_factory=list)
    gestures: list[MPCategory] = field(default_factory=list)
    handedness: list[MPCategory] = field(default_factory=list)

    # Note: Cached variable to speed up repeated invocations
    _bounds: tuple[float,float,float,float] = None

    @staticmethod
    def mpCategoriesStr(mpCategories:list[MPCategory]):
        return ", ".join(
            [f"{mpCategory.category_name}:{mpCategory.score:.2f}" for mpCategory in mpCategories]
        )        

    def gesturesStr(self):
        return self.mpCategoriesStr(self.gestures)
    
    def handednessStr(self):
        return self.mpCategoriesStr(self.handedness)    

    def landmarkCoords(self):
        return np.array([
            [landmark.x, landmark.y]
            for landmark in self.landmarks
        ])

    def landmark(self, landmarkType:MPHandLandmark) -> MPNormalizedLandmark:
        if landmarkType < len(self.landmarks):
            return self.landmarks[landmarkType]
        return None

    def boundingBox(self) -> tuple[float, float, float, float]:
        """
            Returns (x1,y1,x2,y2) bounding box of landmarks
            where x1 <= x2 and y1 <= y2    
        """        
        
        if self._bounds is not None:
            return self._bounds

        if not self.landmarks:
            self._bounds = (0,0,0,0)

        else:
            coords = self.landmarkCoords()

            x1, y1 = np.min(coords, axis=0)
            x2, y2 = np.max(coords, axis=0)

            self._bounds = (x1, y1, x2, y2)
        
        return self._bounds


@dataclass
class HandAnnotations:
    frameId: int = 0
    frameTime: float = 0
    features: list[HandFeatures] = field(default_factory=list)

    def addMPGestureRecognizerResult(self, result:MPGestureRecognizerResult) -> 'HandAnnotations':
        if not result:
            return self

        handFeatureData = zip(result.handedness, result.hand_landmarks, result.gestures)
        self.features+= [
            HandFeatures(
                landmarks = landmarks, 
                gestures = gestures, 
                handedness = handedness
            )
            for handedness, landmarks, gestures in handFeatureData
        ]

        return self
    
    def draw(self, pixels:np.array):

        height, width, depth = pixels.shape

        for feature in self.features:

            # Draw hand skeleton
            mpDrawing.draw_landmarks(
                pixels,
                MakeNormalizedLandmarkList(feature.landmarks),
                mpHands.HAND_CONNECTIONS,
                mpDrawingStyles.get_default_hand_landmarks_style(),
                mpDrawingStyles.get_default_hand_connections_style(),
            )

            # Draw bounding box
            boundingBox = feature.boundingBox()
            x1,y1,x2,y2 = (boundingBox * np.array([width, height, width, height])).astype(int)
            cv.rectangle(pixels, (x1, y1), (x2, y2), color=Color.green)

            # Draw name of gesture above box
            cv.putText(
                pixels,
                feature.gesturesStr(),
                org=(x1, y1),
                fontScale=1,
                color=Color.green,
                fontFace=cv.FONT_HERSHEY_COMPLEX,
            )

            # Draw handedness below box
            cv.putText(
                pixels,
                feature.handednessStr(),
                org=(x1, y2),
                fontScale=1,
                color=Color.green,
                fontFace=cv.FONT_HERSHEY_COMPLEX,
            )



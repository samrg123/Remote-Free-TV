
from util import *

import numpy as np
from typing import Callable



class GestureState:
    name:str

    def __init__(self, name) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

class GestureStates:
    IDLE      = GestureState("IDLE")
    WAITING   = GestureState("WAITING")
    ACTIVATED = GestureState("ACTIVATED")
    COOLDOWN  = GestureState("COOLDOWN")

class Gesture:

    defaultWaitingFrames:int = 5

    def __init__(self, mpGestures:list[str], rokuKey:str, mpHandness:list[str] = ["left", "right"], 
                 rokuCommand:str = "keypress", waitingFrames:int = defaultWaitingFrames) -> None:
        
        self.mpGestures  = mpGestures
        self.rokuKey     = rokuKey
        self.rokuCommand = rokuCommand

        self.waitingFrameId = 0
        self.updateFrameId  = 0 
        self.waitingFrames  = waitingFrames

        self.isReady = self._isReady
        self.isDeactivated = self._isDeactivated

        self.state = GestureStates.IDLE 

    # Note: Virtual functions that can be overridden
    @staticmethod
    def _isReady(self, handAnnotations) -> bool:
        elapsedFrames = self.updateFrameId - self.waitingFrameId
        return elapsedFrames >= self.waitingFrames
    
    @staticmethod
    def _isDeactivated(self, handAnnotations) -> bool:
        # Default action is to immediately deactivate gesture so it only gets detected once        
        return True

    def setState(self, newState) -> None:        

        if self.state == newState:
            return

        log(f"Gesture: {self} | StateChange {self.state} -> {newState}")
        self.state = newState

    def updateState(self, frameId, handAnnotations) -> None:

        self.updateFrameId = frameId

        if handAnnotations is None or len(handAnnotations) == 0:
            self.setState(GestureStates.IDLE)
            return

        # TODO: right now we only look at the first annotation. We really should parse them all
        # TODO: Make sure we sync the detected hand to the same annotation each time
        # TODO: Account for handedness
        landmarks, handedness, gestures = handAnnotations[0] 

        foundGesture = False
        for gesture in gestures:
            if gesture.category_name in self.mpGestures:
                foundGesture = True
                break

        if self.state == GestureStates.IDLE: 
            if foundGesture:
                self.waitingFrameId = frameId
                self.setState(GestureStates.WAITING)

        # Note: Default Action for all cases below
        if not foundGesture:
            self.setState(GestureStates.IDLE)
            return

        if self.state == GestureStates.WAITING: 
            if self.isReady(self, handAnnotations):
                self.setState(GestureStates.ACTIVATED)

            # Note: No fallthrough, always force 1 waiting activated frame to allow for detection
            return

        if self.state == GestureStates.ACTIVATED:
            if self.isDeactivated(self, handAnnotations):
                self.setState(GestureStates.COOLDOWN)


    def isDetected(self, frameId, handAnnotations) -> bool:
        self.updateState(frameId, handAnnotations)
        return self.state == GestureStates.ACTIVATED
    
    def __str__(self) -> str:
        return f"{self.mpGestures} | {self.rokuKey} | {self.rokuCommand} | {self.state}"

class StaticGesture(Gesture):  
  
    def __str__(self) -> str:
        return f"StaticGesture: {super().__str__()}"

class MotionGesture(Gesture):
    defaultDebounceLength:int = 15
    defaultMotionThreshold:float = .2

    def __init__(self, mpGestures: str, rokuKey: str, isReady, rokuCommand: str = "keypress", 
                 debounceLength:int = defaultDebounceLength, motionThreshold:float = defaultMotionThreshold) -> None:
        
        super().__init__(mpGestures, rokuKey, rokuCommand)

        self.debounceLength = debounceLength
        self.motionThreshold = motionThreshold

        self.isReady = isReady

        self.history = np.empty((debounceLength), dtype=object)

    def motionDelta(self) -> tuple[float, float]:
        
        if not self.history[0] or not self.history[-1]:
            return (0, 0)

        xDelta = self.history[-1].x - self.history[0].x
        yDelta = self.history[-1].y - self.history[0].y

        return (xDelta, yDelta)

    # Note: Left and Right motions flipped to account for mirroring of image
   
    def isReadyLeft(self, handAnnotations) -> bool:
        if not self._isReady(self, handAnnotations):
            return False

        xDelta, yDelta = self.motionDelta()
        return abs(xDelta) > abs(yDelta) and -xDelta >= self.motionThreshold
    
    def isReadyRight(self, handAnnotations) -> bool:
        if not self._isReady(self, handAnnotations):
            return False

        xDelta, yDelta = self.motionDelta()
        return abs(xDelta) > abs(yDelta) and xDelta >= self.motionThreshold

    def isReadyUp(self, handAnnotations) -> bool:
        if not self._isReady(self, handAnnotations):
            return False

        xDelta, yDelta = self.motionDelta()
        return abs(yDelta) > abs(xDelta) and yDelta >= self.motionThreshold

    def isReadyDown(self, handAnnotations) -> bool:
        if not self._isReady(self, handAnnotations):
            return False

        xDelta, yDelta = self.motionDelta()
        return abs(yDelta) > abs(xDelta) and -yDelta >= self.motionThreshold    

    def updateState(self, frameId, handAnnotations):

        # Update history
        self.history = np.roll(self.history, 1)  
        self.history[0] = None
        
        # From user's perspective, x,y at 0,0 is in top right, 1,1 is bottom left
        # TODO: Right now we only look at first handAnnotation, we should process all of them!
        if handAnnotations is not None and len(handAnnotations) > 0:    

            landmarks, handedness, gestures = handAnnotations[0]
            if len(landmarks) > 0:
                self.history[0] = landmarks[0]        
        
        super().updateState(frameId, handAnnotations)

    def __str__(self) -> str:
        return f"MotionGesture: {super().__str__()} | {self.isReady}"

class GesturesClass:

    left  = MotionGesture(["fist", "dislike"], "Left",   MotionGesture.isReadyLeft)
    right = MotionGesture(["fist", "dislike"], "Right",  MotionGesture.isReadyRight)
    up    = MotionGesture(["fist", "dislike"], "Up",     MotionGesture.isReadyUp)
    down  = MotionGesture(["fist", "dislike"], "Down",   MotionGesture.isReadyDown)

    rewind      = MotionGesture(["four"], "Rev", MotionGesture.isReadyLeft)
    fastForward = MotionGesture(["four"], "Fwd", MotionGesture.isReadyRight)

    back  = MotionGesture(["two_up_inverted", "peace_inverted"], "Back",   MotionGesture.isReadyLeft)

    home   = StaticGesture(["rock"],            "Home")
    play   = StaticGesture(["peace", "two_up"], "Play")
    select = StaticGesture(["ok"],              "Select")

    def gestureList(self):
        
        gestures = []
        for itemName in dir(self):

            item = getattr(self, itemName)
            if isinstance(item, Gesture):
                gestures.append(item) 

        return gestures

    def __iter__(self):
        return self.gestureList().__iter__()

    # placeholders / unsupported
    # power      = StaticGesture("ok", "Power")
    # speech     = StaticGesture("three2", "Speech")
    # volumeUp   = MotionGesture("one", "VolumeUp",   motionUp)
    # volumeDown = MotionGesture("one", "VolumeDown", motionDown)

Gestures = GesturesClass()
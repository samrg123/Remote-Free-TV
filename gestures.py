
import numpy as np
from typing import Callable

class Gesture:
    
    # TODO: Add handedness!

    def __init__(self, mpName:str, rokuKey:str, rokuCommand:str = "keypress") -> None:
        self.mpGesture   = mpName
        self.rokuKey     = rokuKey
        self.rokuCommand = rokuCommand

    def isDetected():
        """
            Returns true if gesture is detected, false otherwise
        """
        return False
    
    def __str__(self) -> str:
        return f"{self.mpGesture} | {self.rokuKey} | {self.rokuCommand}"

class StaticGesture(Gesture):
    def isDetected(self, handAnnotations):
        for (landmarks, handedness, gestures) in handAnnotations:
            for gesture in gestures:
                if gesture.category_name == self.mpGesture:
                    return True
        return False    
    
    def __str__(self) -> str:
        return f"StaticGesture: {super().__str__()}"

class MotionGesture(Gesture):
    defaultDebounceLength:int = 15

    def __init__(self, mpName: str, rokuKey: str, motionDetected, rokuCommand: str = "keypress", debounceLength = defaultDebounceLength) -> None:
        super().__init__(mpName, rokuKey, rokuCommand)

        self.debounceLength = debounceLength
        self.history = np.empty((debounceLength), dtype=object)

        self.motionDetected = motionDetected

    def isDetected(self, handAnnotations):
        
        # Update history
        self.history = np.roll(self.history, 1)  
        
        if handAnnotations:   

            # From user's perspective, x,y at 0,0 is in top right, 1,1 is bottom left
            foundGesture = False
            for (landmarks, handedness, gestures) in handAnnotations:

                for gesture in gestures:
                    if gesture.category_name == self.mpGesture:
                        self.history[0] = landmarks[0]
                        foundGesture = True
                
                if foundGesture:
                    break
        else:
            self.history[0] = None

        return self.motionDetected(self.history)

    def __str__(self) -> str:
        return f"MotionGesture: {super().__str__()} | {self.motionDetected}"

class GesturesClass:

    def motionLeft(history):
        if history[0] is not None and history[-1] is not None:
            return history[-1].x - history[0].x > 0.2
        return False 

    def motionRight(history):
        if history[0] is not None and history[-1] is not None:
            return history[0].x - history[-1].x > 0.2
        return False
    
    def motionUp(history):
        if history[0] is not None and history[-1] is not None:
            return history[-1].y - history[0].y > 0.2
        return False
    
    def motionDown(history):
        if history[0] is not None and history[-1] is not None:
            return history[0].y - history[-1].y > 0.2
        return False

    left  = MotionGesture("stop",           "Left",   motionLeft)
    right = MotionGesture("stop",           "Right",  motionRight)
    up    = MotionGesture("stop",           "Up",     motionUp)
    down  = MotionGesture("stop",           "Down",   motionDown)
    back  = MotionGesture("peace_inverted", "Back",   motionLeft) #TODO: Make sure this works - ideally we would want sideways thumb

    home   = StaticGesture("call", "Home")
    select = StaticGesture("fist", "Select")
    play   = StaticGesture("peace", "Play")

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
    # speach     = StaticGesture("mute", "Speach")
    # volumeUp   = MotionGesture("one", "VolumeUp",   motionUp)
    # volumeDown = MotionGesture("one", "VolumeDown", motionDown)

Gestures = GesturesClass()
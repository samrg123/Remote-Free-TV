from util import *

import numpy as np
from typing import Callable


class GestureState:
    name: str

    def __init__(self, name) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class GestureStates:
    IDLE = GestureState("IDLE")
    WAITING = GestureState("WAITING")
    ACTIVATED = GestureState("ACTIVATED")
    COOLDOWN = GestureState("COOLDOWN")


class Gesture:
    defaultWaitingTimeSeconds: int = 1.0
    defaultElasticitySeconds = 0.25

    def __init__(
        self,
        mpGestures: list[str],
        rokuKey: str,
        mpHandness: list[str] = ["left", "right"],
        rokuCommand: str = "keypress",
        waitingTimeSeconds: int = defaultWaitingTimeSeconds,
        gestureElasticitySeconds: int = defaultElasticitySeconds,
    ) -> None:
        self.mpGestures = mpGestures
        self.rokuKey = rokuKey
        self.rokuCommand = rokuCommand

        self.waitingFrameId = 0
        self.updateFrameId = 0
        self.waitingTimeSeconds = waitingTimeSeconds

        self.isReady = self._isReady
        self.isDeactivated = self._isDeactivated

        self.state = GestureStates.IDLE
        self.gestureElasticitySeconds = gestureElasticitySeconds
        self.elasticFrameId = 0

    # Note: Virtual functions that can be overridden
    @staticmethod
    def _isReady(self, handAnnotations, frameRate) -> bool:
        elapsedFrames = self.updateFrameId - self.waitingFrameId
        return elapsedFrames >= int(self.waitingTimeSeconds * frameRate)

    @staticmethod
    def _isDeactivated(self, handAnnotations, frameRate) -> bool:
        # Default action is to immediately deactivate gesture so it only gets detected once
        return True

    def setState(self, newState) -> None:
        if self.state == newState:
            return

        # log(f"Gesture: {self} | StateChange {self.state} -> {newState}")
        self.state = newState

    def updateState(self, frameId, handAnnotations, frameRate) -> None:
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
                self.elasticFrameId = frameId

        # Note: Default Action for all cases below
        if not foundGesture:
            if frameId - self.elasticFrameId > int(
                self.gestureElasticitySeconds * frameRate
            ):
                self.setState(GestureStates.IDLE)
            return

        self.elasticFrameId = frameId
        if self.state == GestureStates.WAITING:
            if self.isReady(self, handAnnotations, frameRate):
                self.setState(GestureStates.ACTIVATED)

            # Note: No fallthrough, always force 1 waiting activated frame to allow for detection
            return

        if self.state == GestureStates.ACTIVATED:
            if self.isDeactivated(self, handAnnotations, frameRate):
                self.setState(GestureStates.COOLDOWN)

    def isDetected(self, frameId, handAnnotations, frameRate) -> bool:
        self.updateState(frameId, handAnnotations, frameRate)
        return self.state == GestureStates.ACTIVATED

    def __str__(self) -> str:
        return f"{self.mpGestures} | {self.rokuKey} | {self.rokuCommand} | {self.state}"


class StaticGesture(Gesture):
    def __str__(self) -> str:
        return f"StaticGesture: {super().__str__()}"


class MotionGesture(Gesture):
    defaultDebounceTimeSeconds: float = 1.0
    defaultMotionThreshold: float = 0.2

    def __init__(
        self,
        mpGestures: str,
        rokuKey: str,
        isReady,
        rokuCommand: str = "keypress",
        debounceSeconds: int = defaultDebounceTimeSeconds,
        motionThreshold: float = defaultMotionThreshold,
    ) -> None:
        super().__init__(mpGestures, rokuKey, rokuCommand)

        self.debounceSeconds = debounceSeconds
        self.motionThreshold = motionThreshold

        self.isReady = isReady

        # Be able to hold up to 2 times the needed history at 15 FPS
        self.historyLength = int(debounceSeconds * 15 * 2)
        self.history = np.empty((self.historyLength), dtype=object)

    def motionDelta(self, frameRate) -> tuple[float, float]:
        frameIdxDebounce = min(
            int(self.debounceSeconds * frameRate), self.historyLength - 1
        )

        if not self.history[0] or not self.history[frameIdxDebounce]:
            return (0, 0)

        xDelta = self.history[frameIdxDebounce].x - self.history[0].x
        yDelta = self.history[frameIdxDebounce].y - self.history[0].y

        return (xDelta, yDelta)

    # Note: Left and Right motions flipped to account for mirroring of image

    def isReadyLeft(self, handAnnotations, frameRate) -> bool:
        if not self._isReady(self, handAnnotations, frameRate):
            return False

        xDelta, yDelta = self.motionDelta(frameRate)
        return abs(xDelta) > abs(yDelta) and -xDelta >= self.motionThreshold

    def isReadyRight(self, handAnnotations, frameRate) -> bool:
        if not self._isReady(self, handAnnotations, frameRate):
            return False

        xDelta, yDelta = self.motionDelta(frameRate)
        return abs(xDelta) > abs(yDelta) and xDelta >= self.motionThreshold

    def isReadyUp(self, handAnnotations, frameRate) -> bool:
        if not self._isReady(self, handAnnotations, frameRate):
            return False

        xDelta, yDelta = self.motionDelta(frameRate)
        return abs(yDelta) > abs(xDelta) and yDelta >= self.motionThreshold

    def isReadyDown(self, handAnnotations, frameRate) -> bool:
        if not self._isReady(self, handAnnotations, frameRate):
            return False

        xDelta, yDelta = self.motionDelta(frameRate)
        return abs(yDelta) > abs(xDelta) and -yDelta >= self.motionThreshold

    def updateState(self, frameId, handAnnotations, frameRate):
        # Update history
        self.history = np.roll(self.history, 1)
        self.history[0] = None

        # From user's perspective, x,y at 0,0 is in top right, 1,1 is bottom left
        # TODO: Right now we only look at first handAnnotation, we should process all of them!
        if handAnnotations is not None and len(handAnnotations) > 0:
            landmarks, handedness, gestures = handAnnotations[0]
            if len(landmarks) > 0:
                self.history[0] = landmarks[0]

        super().updateState(frameId, handAnnotations, frameRate)

    def __str__(self) -> str:
        return f"MotionGesture: {super().__str__()} | {self.isReady}"


class GesturesClass:
    left = MotionGesture(["fist", "dislike"], "Left", MotionGesture.isReadyLeft)
    right = MotionGesture(["fist", "dislike"], "Right", MotionGesture.isReadyRight)
    up = MotionGesture(["fist", "dislike"], "Up", MotionGesture.isReadyUp)
    down = MotionGesture(["fist", "dislike"], "Down", MotionGesture.isReadyDown)

    rewind = MotionGesture(["four"], "Rev", MotionGesture.isReadyLeft)
    fastForward = MotionGesture(["four"], "Fwd", MotionGesture.isReadyRight)

    back = MotionGesture(
        ["two_up_inverted", "peace_inverted"], "Back", MotionGesture.isReadyLeft
    )

    home = StaticGesture(["rock"], "Home")
    play = StaticGesture(["peace", "two_up"], "Play")
    select = StaticGesture(["ok"], "Select")

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

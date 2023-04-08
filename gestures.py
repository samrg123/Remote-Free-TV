from util import *
from cvUtil import *

from handAnnotations import *

class GestureState:
    name: str

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
    defaultRokuCommand: str = "keypress"

    # Note: mediapipe assumes image is flipped so "Left" is actually right hand
    defaultMpHandedness:list[str] = ["Left"]
    defaultMpGestureBlacklist: list[str] = []

    defaultWaitingSeconds: float = 0.5
    defaultElasticitySeconds: float = 0.25

    def __init__(
        self,
        mpGestures: list[str],
        rokuKey: str,
        mpHandedness: list[str] = defaultMpHandedness,
        blacklistMpGestures: list[str] = defaultMpGestureBlacklist,
        rokuCommand: str = defaultRokuCommand,
        waitingSeconds: float = defaultWaitingSeconds,
        elasticitySeconds: float = defaultElasticitySeconds
    ) -> None:

        self.mpGestures = mpGestures
        self.mpHandedness = mpHandedness
        self.blacklistMpGestures = blacklistMpGestures

        self.rokuKey = rokuKey
        self.rokuCommand = rokuCommand

        self.updateFrameTime = 0
        self.waitingFrameTime = 0
        self.waitingSeconds = waitingSeconds

        self.lastGestureTime = 0
        self.elasticitySeconds = elasticitySeconds

        self.isReady = self._isReady
        self.isReset = self._isReset
        self.isDeactivated = self._isDeactivated

        self.state = GestureStates.IDLE

    # Note: Virtual functions that can be overridden
    @staticmethod
    def _isReady(self, handAnnotations:HandAnnotations) -> bool:
        elapsedTime = self.updateFrameTime - self.waitingFrameTime
        return elapsedTime >= self.waitingSeconds

    @staticmethod
    def _isReset(self, handAnnotations:HandAnnotations) -> bool:
        elapsedTime = self.updateFrameTime - self.lastGestureTime 
        return elapsedTime >= self.elasticitySeconds

    @staticmethod
    def _isDeactivated(self, handAnnotations:HandAnnotations) -> bool:
        # Default action is to immediately deactivate gesture so it only gets detected once
        return True

    def setState(self, newState:GestureState) -> None:
        if self.state == newState:
            return

        log(f"Gesture: {self} | StateChange {self.state} -> {newState}", logLevel=LogLevel.Debug)
        self.state = newState

    @staticmethod
    def mostLikely(
        handAnnotations:HandAnnotations, 
        handFilter:list[str], 
        gestureFilter:list[str]
    ) -> HandFeatures:
            
        maxGestureScore = -1
        maxFeature = HandFeatures()
        for feature in handAnnotations.features:

            maxHandednessScore = -1
            maxHandedness = []
            for hand in feature.handedness:
                
                if hand.score > maxHandednessScore and \
                    hand.category_name in handFilter:

                    maxHandednessScore = hand.score
                    maxHandedness = hand

            if not maxHandedness:
                continue

            for gesture in feature.gestures:

                if gesture.score > maxGestureScore and \
                    gesture.category_name in gestureFilter:
                    
                    maxGestureScore = gesture.score
                    maxFeature = HandFeatures(
                        landmarks=feature.landmarks, 
                        gestures=[gesture], 
                        handedness=[maxHandedness]
                    )

        return maxFeature

    def mostLikelyMatch(self, handAnnotations:HandAnnotations) -> HandFeatures:
        return self.mostLikely(handAnnotations, handFilter=self.mpHandedness, gestureFilter=self.mpGestures)

    def mostLikelyBlackList(self, handAnnotations:HandAnnotations) -> HandFeatures:
        return self.mostLikely(handAnnotations, handFilter=self.mpHandedness, gestureFilter=self.blacklistMpGestures)

    def updateState(self, handAnnotations:HandAnnotations) -> None:
        self.updateFrameTime = handAnnotations.frameTime

        # TODO: experiment with this... may help cleanup fast movements, but right now things
        #       are working pretty well with a short elasticity
        #       basically we want to see if there is a mostLikelyBlackList
        #       if there is then we need need to force state into COOLDOWN/IDLE mode?
        # mostLikelyBlackList = self.mostLikelyBlackList(handAnnotations)
        # if mostLikelyBlackList.gestures:
        # self.setState(GestureStates.IDLE)
        #   return

        mostLikelyMatch = self.mostLikelyMatch(handAnnotations)
        foundGesture = (len(mostLikelyMatch.gestures) > 0)

        # Warn: There is a lot a nuance on how these conditions fallthrough 
        #       be cautious while editing

        if self.state == GestureStates.IDLE:
            if foundGesture:
                self.waitingFrameTime = handAnnotations.frameTime
                self.setState(GestureStates.WAITING)  

        if foundGesture:
            self.lastGestureTime = handAnnotations.frameTime

            if self.state == GestureStates.WAITING:
                if self.isReady(self, handAnnotations):
                    self.setState(GestureStates.ACTIVATED)

                # Note: No fallthrough, always stay in waiting state until ready and 
                #       force at least 1 activated frame to allow for detection
                return

        if self.state == GestureStates.ACTIVATED:
            if self.isDeactivated(self, handAnnotations):
                self.setState(GestureStates.COOLDOWN)

        if self.isReset(self, handAnnotations):
            self.setState(GestureStates.IDLE)


    def isDetected(self, handAnnotations:HandAnnotations) -> bool:
        self.updateState(handAnnotations)
        return self.state == GestureStates.ACTIVATED

    def __str__(self) -> str:
        return f"{self.mpGestures} | {self.mpHandedness} | {self.rokuKey} | {self.rokuCommand} | {self.state}"


class StaticGesture(Gesture):
    def __str__(self) -> str:
        return f"StaticGesture: {super().__str__()}"


class MotionGesture(Gesture):

    # TODO: make this change with distance AKA size of hand's bounding box?
    defaultMotionThreshold: float = 0.2
    defaultMotionHistoryDuration: float = 2.0

    def __init__(
        self,
        mpGestures: str,
        rokuKey: str,
        isReady,
        mpHandedness: list[str] = Gesture.defaultMpHandedness,
        mpGestureBlacklist: list[str] = Gesture.defaultMpGestureBlacklist,
        rokuCommand: str = Gesture.defaultRokuCommand,
        waitingSeconds: int = Gesture.defaultWaitingSeconds,
        elasticitySeconds: int = Gesture.defaultElasticitySeconds,
        motionThreshold: float = defaultMotionThreshold,
        historyDuration: float = defaultMotionHistoryDuration
    ) -> None:

        super().__init__(
            mpGestures = mpGestures,
            rokuKey = rokuKey,
            mpHandedness = mpHandedness,
            blacklistMpGestures= mpGestureBlacklist,
            rokuCommand = rokuCommand,
            waitingSeconds = waitingSeconds,
            elasticitySeconds = elasticitySeconds
        )

        self.isReady = isReady

        self.motionThreshold = motionThreshold
        self.historyDuration = historyDuration

        self.history = []

    def resetHistory(self) -> None:
        self.history = self.history[-1:]
        log(f"Reset history: {self}", logLevel=LogLevel.Debug)

    def mostLikelyMatch(self, handAnnotations:HandAnnotations) -> HandFeatures:
        mostLikelyMatch = super().mostLikelyMatch(handAnnotations)

        # Note: we cache the results in handAnnotations so we can continually refer back to them
        # Note: we handAnnotations to store cached result so garbage collection automatically cleans
        #       up after us the handAnnotation is removed from history
        if not hasattr(handAnnotations, "mostLikelyMatch"):
            handAnnotations.mostLikelyMatch = {}
        handAnnotations.mostLikelyMatch[self] = mostLikelyMatch
        
        return mostLikelyMatch

    def setState(self, newState: GestureState) -> None:

        if newState == GestureStates.WAITING or newState == GestureStates.ACTIVATED:
            self.resetHistory()

        super().setState(newState)

    def motionDelta(self, handAnnotations:HandAnnotations) -> tuple[float, float]:

        currentMostLikelyMatch:HandFeatures = handAnnotations.mostLikelyMatch[self]
        
        currentMostLikelyWristLandmark = currentMostLikelyMatch.landmark(MPHandLandmark.WRIST)
        if currentMostLikelyWristLandmark is None:
            return (0,0)
        
        maxDistSquard = 0
        maxDxDy = (0, 0)
        for annotation in self.history:

            mostLikelyMatch:HandFeatures = annotation.mostLikelyMatch[self]
            wristLandmark = mostLikelyMatch.landmark(MPHandLandmark.WRIST)
            if wristLandmark is None:
                continue

            
            dx = wristLandmark.x - currentMostLikelyWristLandmark.x
            dy = wristLandmark.y - currentMostLikelyWristLandmark.y
            
            distSquared = dx*dx + dy*dy

            if distSquared > maxDistSquard:
                maxDistSquard = distSquared
                maxDxDy = (dx, dy)
        
        return maxDxDy

    def isReadyXY(self, handAnnotations:HandAnnotations, deltaIndex:int, deltaSign:int) -> bool:
        if not self._isReady(self, handAnnotations):
            return False

        deltas = self.motionDelta(handAnnotations)

        absDeltas = np.abs(deltas)
        maxAbsDelta = np.max(absDeltas)

        delta = deltas[deltaIndex]*deltaSign
        absDelta = absDeltas[deltaIndex]

        if absDelta == maxAbsDelta and delta >= self.motionThreshold:
            return True

        # Went too far from center - reset history so we don't activate 
        if maxAbsDelta > self.motionThreshold:
            log(f"Exceed motionThreshold in wrong direction. Resetting: {maxAbsDelta}", logLevel=LogLevel.Debug)
            self.resetHistory()

        return False

    # Note: Left and Right motions flipped to account for mirroring of image
    def isReadyLeft(self, handAnnotations:HandAnnotations) -> bool:
        return self.isReadyXY(handAnnotations, deltaIndex=0, deltaSign=-1)

    def isReadyRight(self, handAnnotations:HandAnnotations) -> bool:
        return self.isReadyXY(handAnnotations, deltaIndex=0, deltaSign=1)

    def isReadyUp(self, handAnnotations:HandAnnotations) -> bool:
        return self.isReadyXY(handAnnotations, deltaIndex=1, deltaSign=1)

    def isReadyDown(self, handAnnotations:HandAnnotations) -> bool:
        return self.isReadyXY(handAnnotations, deltaIndex=1, deltaSign=-1)        

    def updateState(self, handAnnotations:HandAnnotations):

        # find last expired index
        # TODO: replace this with binary search
        nonExpiredHistoryIndex = len(self.history)
        for i in range(len(self.history)):
            oldAnnotation = self.history[i]
            ellapsedTime = handAnnotations.frameTime - oldAnnotation.frameTime
            if ellapsedTime <= self.historyDuration:
                nonExpiredHistoryIndex = i
                break

        # update history chopping off expired indices
        self.history = self.history[nonExpiredHistoryIndex:] + [handAnnotations]        
        super().updateState(handAnnotations)

    def __str__(self) -> str:
        return f"MotionGesture: {super().__str__()} | {self.isReady.__name__}"


class GesturesClass:
    left        = MotionGesture(["fist", "dislike"], "Left",  MotionGesture.isReadyLeft,  mpGestureBlacklist=["palm", "stop"])
    right       = MotionGesture(["fist", "dislike"], "Right", MotionGesture.isReadyRight, mpGestureBlacklist=["palm", "stop"])
    up          = MotionGesture(["fist", "dislike"], "Up",    MotionGesture.isReadyUp,    mpGestureBlacklist=["palm", "stop"])
    down        = MotionGesture(["fist", "dislike"], "Down",  MotionGesture.isReadyDown,  mpGestureBlacklist=["palm", "stop"])

    rewind      = MotionGesture(["four"], "Rev", MotionGesture.isReadyLeft )
    fastForward = MotionGesture(["four"], "Fwd", MotionGesture.isReadyRight)

    back = MotionGesture(
        ["two_up_inverted", "peace_inverted"], "Back", MotionGesture.isReadyLeft
    )

    home   = StaticGesture(["rock"], "Home")
    play   = StaticGesture(["peace", "two_up"], "Play")
    select = StaticGesture(["ok"], "Select")

    def gestureList(self) -> list[Gesture]:
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

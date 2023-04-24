import numpy as np
from attr import dataclass

@dataclass
class Frame:
    id:int = 0
    time:float = 0
    frameRate:float = 0
    
    colorPixels:np.array = None
    depthPixels:np.array = None

    def timeMs(self) -> int:
        return int(self.time*1000)
    

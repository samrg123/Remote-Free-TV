from dataclasses import dataclass
import cv2 as cv

# NOTE: Colors are in BGR to align with opencv
class Color:
    
    Type = tuple[int, int, int]

    white = (255, 255, 255)
    black = (  0,   0,   0)
    red   = (  0,   0, 255)
    green = (  0, 255,   0)
    blue  = (255,   0,   0)

@dataclass
class CvText:
    text : str
    origin : tuple[int, int]

    font : int = cv.FONT_HERSHEY_DUPLEX
    fontColor : Color.Type = Color.white
    fontSize : float = .75
    fontThickness : int = 1

    shadowColor : Color.Type = Color.black
    shadowSize : float = 2.0
    origin : tuple[int, int] = (0, 0)

    def getShadowThickness(self) -> int:
        return int(self.fontThickness + self.shadowSize + 0.5)

    def getSize(self) -> tuple[int, int]:
        textWidth, textHeight  = cv.getTextSize(self.text, self.font, self.fontSize, self.getShadowThickness())[0]
        return (textWidth, textHeight)
    
    def draw(self, pixels:cv.Mat) -> cv.Mat:

        # Note: safe guard in case values get set to float which would cause openCV to throw an exception  
        intOrigin = (int(self.origin[0]), int(self.origin[1]))
        intFontThickness = int(self.fontThickness)


        shadowImage = cv.putText(pixels, self.text, intOrigin, self.font, self.fontSize, self.shadowColor, self.getShadowThickness(), cv.LINE_AA)
        return cv.putText(shadowImage, self.text, intOrigin, self.font, self.fontSize, self.fontColor, intFontThickness, cv.LINE_AA)    
    
class NamedImage:
    defaultNamePadding:tuple[int,int] = [10, 10]

    def __init__(self, name:str, pixels:cv.Mat, nameOrigin:tuple[int, int] = None) -> None:

        self.cvText = CvText(name)
        self.pixels = pixels
        
        if nameOrigin is None:
            _, nameHeight  = self.cvText.getSize()
            self.cvText.origin = (self.defaultNamePadding[0], nameHeight + self.defaultNamePadding[1])
        
        else:
            self.cvText.origin = nameOrigin

    def getImage(self) -> cv.Mat:
        return self.cvText.draw(self.pixels)

def listVideoPorts(maxNonworkingPorts = 5):
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    
    available_ports   = []
    working_ports     = []
    non_working_ports = []
    
    # TODO: We really should just list /dev/video* and iterate those devices
    dev_port = 0
    while len(non_working_ports) <= maxNonworkingPorts: 

        camera = cv.VideoCapture(dev_port)

        if not camera.isOpened():

            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)

        else:

            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)

            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)

            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)

        dev_port +=1

    return available_ports, working_ports, non_working_ports
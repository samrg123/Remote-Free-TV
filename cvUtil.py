import cv2 as cv

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
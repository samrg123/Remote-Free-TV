import argparse
import requests
from gestures import Gesture

commandList = [
    "keyup",
    "keydown",
    "keypress",
]

keyList = [
    "Home",
    "Rev",
    "Fwd",
    "Play",
    "Select",
    "Left",
    "Right",
    "Down",
    "Up",
    "Back",
    "InstantReplay",
    "Info",
    "Backspace",
    "Search",
    "Enter"
]

keyDict = {
    Gesture.RightToLeft: "Left",
    Gesture.LeftToRight: "Right",
}


defaultUrl = "http://192.168.4.24:8060"
defaultCmd = "keypress"

class RokuECP():
    def __init__(self, url, defaultCmd):
        self.url = url
        self.defaultCmd = defaultCmd

    def sendGesture(self, gesture, command = None):
        if gesture is not Gesture.Nothing:
            self.sendCommand(keyDict[gesture], command)

    def sendCommand(self, key, command = None):
        command = command if command is not None else self.defaultCmd
        if command not in commandList:
            self.panic(f"Command: '{command}' is not in commandList: {commandList}")
        
        if key not in keyList:
            self.panic(f"Key: '{key}' is not in keyList: {keyList}")

        ecpUrl = f"{self.url}/{command}/{key}"

        try:
            requests.post(ecpUrl)

        except Exception as e:
            self.panic(f"Failed to send ecpUrl: {ecpUrl} | Exception: {e}")

    def panic(self, msg, errorCode = 1):

        print(f"ERROR - {msg}")
        exit(errorCode)

def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--url",     required=False, action="store", metavar="URL", default=defaultUrl, help=f"Sets the url for connecting to the Roku. Default: '{defaultUrl}'")
    argParser.add_argument("--command", required=False, action="store", metavar="CMD", default=defaultCmd, help=f"Sets the command to send to the Roku. Default: '{defaultCmd}' | Values: {commandList}")
    argParser.add_argument("key",         action="store", metavar="KEY", help=f"The key code to send to the Roku. Values: {keyList}")

    args = argParser.parse_args()
    RokuECPInstance = RokuECP(args.url, args.command)
    RokuECPInstance.sendGesture(Gesture.LeftToRight)

if __name__ == "__main__":
    main()
import argparse
import requests
from gestures import Gesture

from util import *

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

# TODO: Clean this up!
keyDict = {
    Gesture.RightToLeft: "Left",
    Gesture.LeftToRight: "Right",
}

class RokuECP():

    defaultUrl:str = "http://192.168.4.24:8060"
    defaultCmd:str = "keypress"

    def __init__(self, url = defaultUrl, defaultCmd = defaultCmd):
        self.url = url
        self.defaultCmd = defaultCmd

    def sendGesture(self, gesture, command = None):
        if gesture is not Gesture.Nothing:
            self.sendCommand(keyDict[gesture], command)

    def sendCommand(self, key, command = None):

        command = command if command is not None else self.defaultCmd 
        if command not in commandList:
            error(f"Command: '{command}' is not in commandList: {commandList}")
            return

        if key not in keyList:
            error(f"Key: '{key}' is not in keyList: {keyList}")
            return

        ecpUrl = f"{self.url}/{command}/{key}"

        try:
            log(f"Sending ecpUrl: {ecpUrl}")
            requests.post(ecpUrl)

        except Exception as e:
            error(f"Failed to send ecpUrl: {ecpUrl} | Exception: {e}")

def main():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--url",     required=False, action="store", metavar="URL", default=RokuECP.defaultUrl, help=f"Sets the url for connecting to the Roku. Default: '{RokuECP.defaultUrl}'")
    argParser.add_argument("--command", required=False, action="store", metavar="CMD", default=RokuECP.defaultCmd, help=f"Sets the command to send to the Roku. Default: '{RokuECP.defaultCmd}' | Values: {commandList}")
    argParser.add_argument("key",       action="store", metavar="KEY", help=f"The key code to send to the Roku. Values: {keyList}")

    args = argParser.parse_args()
    RokuECPInstance = RokuECP(args.url)

    RokuECPInstance.sendCommand(args.key, command=args.command)

if __name__ == "__main__":
    main()
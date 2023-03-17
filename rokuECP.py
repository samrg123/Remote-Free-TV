import argparse
import requests

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


defaultUrl = "http://192.168.0.107:8060"
defaultCmd = "keypress"

argParser = argparse.ArgumentParser()
argParser.add_argument("--url",     required=False, action="store", metavar="URL", default=defaultUrl, help=f"Sets the url for connecting to the Roku. Default: '{defaultUrl}'")
argParser.add_argument("--command", required=False, action="store", metavar="CMD", default=defaultCmd, help=f"Sets the command to send to the Roku. Default: '{defaultCmd}' | Values: {commandList}")
argParser.add_argument("key",         action="store", metavar="KEY", help=f"The key code to send to the Roku. Values: {keyList}")


def Panic(msg, errorCode = 1):

    print(f"ERROR - {msg}")
    exit(errorCode)

def main():

    args = argParser.parse_args()

    if args.command not in commandList:
        Panic(f"Command: '{args.command}' is not in commandList: {commandList}")
    
    if args.key not in keyList:
        Panic(f"Key: '{args.key}' is not in keyList: {keyList}")

    ecpUrl = f"{args.url}/{args.command}/{args.key}"

    try:
        requests.post(ecpUrl)

    except Exception as e:
        Panic(f"Failed to send ecpUrl: {ecpUrl} | Exception: {e}")

if __name__ == "__main__":
    main()
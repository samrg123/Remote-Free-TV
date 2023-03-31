import argparse
import requests

from util import *


class RokuValue:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class RokuECP:
    defaultUrl: str = "http://192.168.4.24:8060"
    defaultCmd: str = "keypress"

    commands: list[str] = ["keyup", "keydown", "keypress"]

    keys: list[str] = [
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
        "Enter",
    ]

    def __init__(self, url=defaultUrl, defaultCmd=defaultCmd):
        assert defaultCmd in self.commands

        self.url = url
        self.defaultCmd = defaultCmd

    def sendCommand(self, key, command=None):
        command = command if command is not None else self.defaultCmd
        if command not in self.commands:
            error(f"Command: '{command}' is not in commandList: {self.commands}")
            return

        if key not in self.keys:
            error(f"Key: '{key}' is not in keyList: {self.keys}")
            return

        ecpUrl = f"{self.url}/{command}/{key}"

        try:
            log(f"Sending ecpUrl: {ecpUrl}")
            requests.post(ecpUrl)

        except Exception as e:
            error(f"Failed to send ecpUrl: {ecpUrl} | Exception: {e}")


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--url",
        required=False,
        action="store",
        metavar="URL",
        default=RokuECP.defaultUrl,
        help=f"Sets the url for connecting to the Roku. Default: '{RokuECP.defaultUrl}'",
    )
    argParser.add_argument(
        "--command",
        required=False,
        action="store",
        metavar="CMD",
        default=RokuECP.defaultCmd,
        help=f"Sets the command to send to the Roku. Default: '{RokuECP.defaultCmd}' | Values: {RokuECP.commands}",
    )
    argParser.add_argument(
        "key",
        action="store",
        metavar="KEY",
        help=f"The key code to send to the Roku. Values: {RokuECP.keys}",
    )

    args = argParser.parse_args()
    RokuECPInstance = RokuECP(args.url)

    RokuECPInstance.sendCommand(args.key, command=args.command)


if __name__ == "__main__":
    main()

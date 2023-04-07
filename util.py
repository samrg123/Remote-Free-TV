class LogLevel:
    class Type:

        name: str
        value: int

        def __init__(self, name, value) -> None:
            self.name = name
            self.value = value

        def __str__(self) -> str:
            return f"{self.name} = {self.value}"
        
        def __lt__(self, logLevelType):
            return self.value < logLevelType.value
                
        def __le__(self, logLevelType):
            return self.value <= logLevelType.value

        def __gt__(self, logLevelType):
            return self.value > logLevelType.value
        
        def __ge__(self, logLevelType):
            return self.value >= logLevelType.value
                
        def __eq__(self, logLevelType):
            return self.value == logLevelType.value
        
        def __ne__(self, logLevelType):
            return self.value != logLevelType.value


    Debug = Type("Debug", 2)
    Warn  = Type("Warn",  1)
    Error = Type("Error", 0) 
    NONE  = Type("NONE", -1) 

    # TODO: This code is similar to gestures.py ... abstract out into generic class
    @staticmethod
    def logLevelList() -> list[Type]:
        logLevels = []
        for itemName in dir(LogLevel):
            item = getattr(LogLevel, itemName)
            if isinstance(item, LogLevel.Type):
                logLevels.append(item)

        return logLevels
    
    @staticmethod
    def fromValue(value:int) -> Type:

        for level in LogLevel.logLevelList():
            if level.value == value:
                return level 
        
        return LogLevel.Type("Unknown", value) 

verboseLevel: int = LogLevel.Warn

def setVerboseLevel(logLevel:LogLevel.Type) ->None:
    global verboseLevel
    verboseLevel = logLevel

def log(msg, prefix: str = "MSG", logLevel: LogLevel.Type = LogLevel.Debug):
    if verboseLevel >= logLevel:
        print(f"{prefix} - {msg}")


def warn(msg):
    log(msg, prefix="WARN", logLevel=LogLevel.Warn)


def error(msg):
    log(msg, prefix="ERROR", logLevel=LogLevel.Error)


def panic(msg, exitCode=1):
    log(msg, prefix="PANIC", logLevel=LogLevel.Error)
    exit(exitCode)

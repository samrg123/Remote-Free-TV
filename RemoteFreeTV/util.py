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


    Debug   = Type("Debug",   3)
    Verbose = Type("Verbose", 2)
    Warn    = Type("Warn",    1)
    Error   = Type("Error",   0) 
    NONE    = Type("NONE",   -1) 

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

# Note: global variables are scoped to python modules, so we must control
#       the verbose level using getters and setters 
def getVerboseLevel() -> LogLevel.Type:
    return _verboseLevel

def setVerboseLevel(logLevel:LogLevel.Type) -> LogLevel.Type:
    """ Sets to global verbose level. Returns the newly assigned verboseLevel"""
    global _verboseLevel
    _verboseLevel = logLevel
    return _verboseLevel

# Note: We don't assign directly to prevent it from being imported
setVerboseLevel(LogLevel.Verbose)

def log(msg, prefix: str = "MSG", logLevel: LogLevel.Type = LogLevel.Verbose):
    if _verboseLevel >= logLevel:
        print(f"{prefix} - {msg}")

def warn(msg):
    log(msg, prefix="WARN", logLevel=LogLevel.Warn)


def error(msg):
    log(msg, prefix="ERROR", logLevel=LogLevel.Error)


def panic(msg, exitCode=1):
    log(msg, prefix="PANIC", logLevel=LogLevel.Error)
    exit(exitCode)

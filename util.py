class LogLevel:
    Debug = 2
    Warn  = 1
    Error = 0

verboseLevel:int = LogLevel.Debug

def log(msg, prefix:str = "MSG", logLevel:int = LogLevel.Debug):
    if(verboseLevel >= logLevel):
        print(f"{prefix} - {msg}")

def warn(msg):
    log(msg, prefix="WARN", logLevel=LogLevel.Warn)

def error(msg):
    log(msg, prefix="ERROR", logLevel=LogLevel.Error)

def panic(msg, exitCode=1):
    log(msg, prefix="PANIC", logLevel=LogLevel.Error)
    exit(exitCode)
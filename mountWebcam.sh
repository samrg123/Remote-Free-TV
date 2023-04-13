#! /bin/bash

# webcamBusId=1-12 #Integrated webcam
webcamBusId=1-2 #Logitech C920
# webcamBusId=7-3 #HD Lifecam

helpCommand=--help
unmountCommand=--unmount

if [ "$1" == "$helpCommand" ]; then
    echo Invoke \'$0\' to attach webcam at bus ID: $webcamBusId   
    echo Invoke \'$0 $unmountCommand\' to detach webcam at bus ID: $webcamBusId   
    echo Invoke \'$0 $helpCommand\' to show this help menu   
    exit 0

elif [ "$1" == "$unmountCommand" ]; then
    command=detach
    echo Detaching USB Bus ID: $webcamBusId to WSL instance... 
else
    command=attach
    echo Attaching USB Bus ID: $webcamBusId to WSL instance...
fi

/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \$p=Start-Process powershell -Verb runAs -PassThru -Wait -ArgumentList \"-c usbipd wsl $command --busid $webcamBusId\"\; Write-Host ExitCode: \$p.ExitCode

if [ "$command" == "attach" ]; then

    # Give Linux sometime to mount webcam to /dev/video*
    echo "Waiting..."
    sleep 2

    echo Giving all users access to: \'`ls /dev/video*`\'
    sudo chmod 777 /dev/video*

    ls -la /dev/video*
fi
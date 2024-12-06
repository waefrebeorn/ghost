@echo off
REM Run script for Ghost Detector

echo ===============================
echo       Ghost Detector Runner
echo ===============================
echo.

REM Activate virtual environment
CALL ghost_env\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment. Ensure the setup.bat script was run successfully.
    pause
    exit /b 1
)

REM Run the Ghost Detector
echo Starting the Ghost Detector...
python ghost_detector.py
IF %ERRORLEVEL% NEQ 0 (
    echo An error occurred while running the Ghost Detector. Check the console for details.
    pause
    exit /b 1
)

REM Deactivate virtual environment after execution
echo Exiting and deactivating virtual environment...
CALL ghost_env\Scripts\deactivate.bat

echo.
echo Ghost Detector has exited.
pause

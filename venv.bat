@echo off
setlocal

:: Check if virtual environment exists, if not, create it
if not exist "ghost_env\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv ghost_env
)

:: Activate the virtual environment
call ghost_env\Scripts\activate

:: Inform the user that the environment is active and provide a command prompt
echo Virtual environment activated. Type your commands below.

:: Open command prompt for user to type commands
cmd /K
@echo off
REM Ghost Detector Setup Script - Improved
echo ===============================
echo   Ghost Detector Setup Script
echo ===============================
echo.

REM Check if Python 3.10 is installed
python --version 2>nul | findstr /C:"Python 3.10" >nul
IF %ERRORLEVEL% NEQ 0 (
    echo Python 3.10 is not installed. Please install Python 3.10 from https://www.python.org/downloads/release/python-3100/ and try again.
    pause
    exit /b 1
) ELSE (
    echo Python 3.10 detected.
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv ghost_env
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)
echo Virtual environment 'ghost_env' created.

REM Activate virtual environment
CALL ghost_env\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated.

REM Upgrade pip
python -m pip install --upgrade pip
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)
echo Pip upgraded.

REM Install PyTorch with CUDA 12.4 support
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu124
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install PyTorch with CUDA 12.4. Trying CPU-only version...
    python -m pip install torch torchvision torchaudio
    IF %ERRORLEVEL% NEQ 0 (
        echo Failed to install PyTorch. Exiting...
        pause
        exit /b 1
    )
) ELSE (
    echo PyTorch with CUDA 12.4 support installed.
)

REM Install other dependencies
python -m pip install transformers tqdm packaging regex safetensors requests pyyaml librosa pydub simpleaudio sounddevice scikit-image mediapipe opencv-python pillow TTS==0.10.0
IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies. Please check logs.
    pause
    exit /b 1
)

REM Download necessary models
python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; GPT2LMHeadModel.from_pretrained('gpt2'); GPT2Tokenizer.from_pretrained('gpt2')" || (
    echo Failed to download GPT-2 model.
    pause
    exit /b 1
)
python -c "from TTS.api import TTS; TTS(model_name='tts_models/en/ljspeech/tacotron2-DCA')" || (
    echo Failed to download TTS model.
    pause
    exit /b 1
)

REM Deactivate virtual environment
CALL ghost_env\Scripts\deactivate.bat
echo Virtual environment deactivated.

echo.
echo Setup completed successfully!
echo To activate the environment, run:
echo     CALL ghost_env\Scripts\activate.bat
echo To start the Ghost Detector, run:
echo     python ghost_detector.py
echo.
pause

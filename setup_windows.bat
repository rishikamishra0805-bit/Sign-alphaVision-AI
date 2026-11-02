@echo off
title SignVision AI - Windows Setup
cd /d "%~dp0"
echo ==========================================
echo       SIGNVISION AI - FIRST TIME SETUP
echo ==========================================
py -3.11 --version
if errorlevel 1 (
  echo Python 3.11 was not found.
  echo Install Python 3.11 64-bit and tick "Add Python to PATH".
  pause
  exit /b 1
)
py -3.11 -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check internet and Python version.
  pause
  exit /b 1
)
echo.
echo Setup completed. Double-click run_windows.bat to start the app.
pause

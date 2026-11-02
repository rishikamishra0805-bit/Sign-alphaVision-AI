@echo off
title SignVision AI
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Please run setup_windows.bat first.
  pause
  exit /b 1
)
call venv\Scripts\activate
python app.py
pause

@echo off
setlocal enabledelayedexpansion
title SSP Vision System
color 0B

echo ===================================================
echo        SSP VISION SYSTEM - STARTUP SEQUENCE
echo ===================================================
echo.
echo [1/3] Checking system environment...
py -3.11 --version >nul 2>&1
if %errorLevel% neq 0 (
    color 4F
    echo.
    echo [ERROR] Python 3.11 is not installed or not in PATH!
    echo Please install Python 3.11 and ensure "Add to PATH" is checked.
    echo.
    echo Please take a photo of this screen and send it to support.
    pause
    exit /b
)

echo [2/4] Verifying and installing missing dependencies...
echo (This may take a moment if it's the first time running)
venv\Scripts\python.exe -m pip install -r requirements.txt -q

echo.
echo [3/4] Preparing User Interface...
:: Start a background timer to open the browser after 5 seconds
start /b powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://localhost:5000'"

echo [3/3] Launching AI Backend and Web Server...
echo.
echo ---------------------------------------------------
echo DO NOT CLOSE THIS WINDOW. 
echo The system is running as long as this window is open.
echo ---------------------------------------------------
echo.

:: Run the application in this exact window so errors are visible
venv\Scripts\python.exe app.py

:: If we reach this line, the application was closed or it crashed.
if %errorLevel% neq 0 (
    color 4F
    echo.
    echo ===================================================
    echo                   CRITICAL ERROR
    echo ===================================================
    echo The SSP Vision System encountered an error and stopped!
    echo.
    echo Please scroll up to read the specific error message above.
    echo Take a photo/screenshot of this entire window and send it to support.
    echo ===================================================
    echo.
    pause
) else (
    echo.
    echo System shut down normally.
    timeout /t 3 >nul
)

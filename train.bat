@echo off
setlocal enabledelayedexpansion

echo Starting AI Job Displacement Model Training...
echo.

cd /d "e:\archive (5)\ai-job-displacement-analysis"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt

echo.
echo ========================================
echo Running Model Training Pipeline...
echo ========================================
echo.

python scripts/train_pipeline.py

if errorlevel 1 (
    echo.
    echo Training failed!
    pause
    exit /b 1
) else (
    echo.
    echo Training completed successfully!
    pause
)

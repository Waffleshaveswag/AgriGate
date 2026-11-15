@echo off
rem AgriGate: double-click to start.
rem First run: installs Python if needed (Windows 10/11), then the model libraries (needs internet).
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (
  where python >nul 2>nul && python -c "import sys" >nul 2>nul && set PY=python
)
if not defined PY (
  echo Python 3 is not installed. Installing it now...
  where winget >nul 2>nul
  if errorlevel 1 (
    echo Could not install automatically. Download Python from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" while installing. Then double-click this file again.
    start "" https://www.python.org/downloads/
    pause
    exit /b 1
  )
  winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
  echo.
  echo Python is installed. Close this window and double-click Start_AgriGate_Windows.bat again.
  pause
  exit /b 0
)
echo Checking the model libraries (first run can take a minute)...
%PY% -m pip install --quiet --disable-pip-version-check -r model\requirements.txt
if errorlevel 1 echo Could not install the libraries (no internet?). The site still opens, using its simpler built-in ranking.
echo Starting AgriGate... your browser will open in a moment.
%PY% start_agrigate.py
pause

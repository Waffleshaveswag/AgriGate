#!/bin/bash
# AgriGate: double-click to start.
# First run: installs the model libraries (needs internet). Your browser opens automatically.
cd "$(dirname "$0")" || exit 1
if ! python3 -c "import sys" >/dev/null 2>&1; then
  echo "Python 3 is not installed. Opening the download page: install it, then double-click this file again."
  open "https://www.python.org/downloads/"
  read -r -p "Press Enter to close..."
  exit 1
fi
echo "Checking the model libraries (first run can take a minute)..."
python3 -m pip install --quiet --disable-pip-version-check -r model/requirements.txt 2>/dev/null \
  || python3 -m pip install --quiet --disable-pip-version-check --user -r model/requirements.txt 2>/dev/null \
  || python3 -m pip install --quiet --disable-pip-version-check --user --break-system-packages -r model/requirements.txt \
  || echo "Could not install the libraries (no internet?). The site still opens, using its simpler built-in ranking."
echo "Starting AgriGate... your browser will open in a moment."
python3 start_agrigate.py

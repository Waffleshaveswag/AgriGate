#!/bin/bash
# AgriGate for Mac: right-click this file -> Open. It runs start_agrigate.py, which installs
# the model libraries on the first run (needs internet) and opens the website in your browser.
cd "$(dirname "$0")" || exit 1
if ! python3 -c "import sys" >/dev/null 2>&1; then
  echo "Python 3 is not installed. Opening the download page: install it, then open this file again."
  open "https://www.python.org/downloads/"
  read -r -p "Press Enter to close..."
  exit 1
fi
python3 start_agrigate.py

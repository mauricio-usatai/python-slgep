#!/bin/bash

set -e

if [ ! -d "build" ]; then
    mkdir build && cp test.py build/
fi

# Clone pybind
if [ ! -d pybind11 ]; then
    git clone https://github.com/pybind/pybind11.git
fi

cd build
# Build lib
cmake .. && make
# Test
python3 -m venv "venv"
source venv/bin/activate
pip install -r ../requirements.txt
python3 test.py
deactivate

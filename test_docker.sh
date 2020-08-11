#!/usr/bin/env bash

source /usr/local/etc/profile.d/conda.sh
conda activate eNNclave
export ENNCLAVE_HOME=/eNNclave

(cd backend/native/tests && python generate_tests.py)
(cd build && cmake ..)
(cd build && make -j8 core_tests && ./core_tests)
(cd frontend/python &&  pip install . && python -m unittest discover)
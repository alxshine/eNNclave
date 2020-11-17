#!/usr/bin/env bash

source /usr/local/etc/profile.d/conda.sh
conda activate eNNclave
source /opt/intel/sgxsdk/environment

# eNNclave setup
export ENNCLAVE_HOME="/eNNclave"
source /opt/intel/sgxsdk/environment

export LD_LIBRARY_PATH="${ENNCLAVE_HOME}/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${ENNCLAVE_HOME}/lib:$PYTHONPATH"

export CC=gcc
export CXX=g++

#(cd backend/native/tests && python generate_tests.py)
(cd build && cmake ..)
#(cd build && make -j8 core_tests && ./core_tests)
(cd build && make backend_sgx_encryptor)
(cd frontend/python &&  pip install . && python -m unittest discover -f)
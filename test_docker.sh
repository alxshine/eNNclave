#!/usr/bin/env bash

source /usr/local/etc/profile.d/conda.sh
conda activate eNNclave
source /opt/intel/sgxsdk/environment
export ENNCLAVE_HOME=/eNNclave
export LD_LIBRARY_PATH=/eNNclave/lib:$LD_LIBRARY_PATH

#(cd backend/native/tests && python generate_tests.py)
(cd build && cmake ..)
#(cd build && make -j8 core_tests && ./core_tests)
(cd build && make backend_sgx_encryptor)
(cd frontend/python &&  pip install . && python -m unittest discover -f)
#!/bin/bash

set -e

# SGX environment setup
source /opt/intel/sgxsdk/environment
# lib path setup
source setup_ld_path.sh

make
python3 build_enclave.py models/amazon.h5 2
python3 time_enclave.py models/amazon_enclave.h5 2
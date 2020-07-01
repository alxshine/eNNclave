#!/bin/bash

set -e

# SGX environment setup
source /opt/intel/sgxsdk/environment
# lib path setup
source setup_ld_path.sh

make interop
python3 build_enclave.py models/mnist.h5 2
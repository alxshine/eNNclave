#!/bin/bash

set -e

# SGX environment setup
source /opt/intel/sgxsdk/environment
# lib path setup
source setup_ld_path.sh

cd lib
make test
./test_matutil
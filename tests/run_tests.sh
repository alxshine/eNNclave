#!/bin/bash

set -e

# SGX environment setup
source /opt/intel/sgxsdk/environment
# lib path setup
source setup_ld_path.sh

# test C code
(cd lib && make test && ./test_matutil)

# test python code
make interop >/dev/null 2>&1 && python3 -m unittest discover
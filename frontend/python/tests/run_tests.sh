#!/bin/bash

set -e

# SGX environment setup
source /opt/intel/sgxsdk/environment
# backend path setup
source setup_ld_path.sh

# tests C code
(cd lib && make test && ./test_matutil)

# tests python code
make interop >/dev/null 2>&1 && python3.8 -m unittest discover
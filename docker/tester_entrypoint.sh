#!/bin/bash

source /opt/intel/sgxsdk/environment
source /eNNclave/setup/setup_ld_path.sh

exec "$@"
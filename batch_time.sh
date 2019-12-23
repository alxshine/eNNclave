#!/bin/bash

for cut in 1 3 5 10 14 18 20 24
# for cut in 1 3
do
  python build_enclave.py models/mit67.h5 $cut
  python time_enclave.py models/mit67_enclave.h5 $cut
done

#!/bin/bash

for cut in {1..24}
# for cut in 1 3 5 10 14 18 21 24
do
  make clean
  make

  python build_enclave.py models/mit67.h5 $cut
  python time_enclave.py models/mit67_enclave.h5 $cut
done

cat timing_logs/mit67_times.csv | mail -s "timing done" alexander.schloegl@uibk.ac.at

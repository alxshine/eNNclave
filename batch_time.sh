#!/bin/bash

# generate pure tf time
python time_enclave.py models/mit.h5 0

for cut in {1..24}
# for cut in 1 3 5 10 14 18 21 24
do
  make clean
  make

  python build_enclave.py models/mit.h5 $cut
  python time_enclave.py models/mit_enclave.h5 $cut
done

cat timing_logs/mit_times.csv | mail -s "timing done" alexander.schloegl@uibk.ac.at

#!/bin/bash

runs_per_index=5

make clean
make

# generate pure tf time
for i in $(seq $runs_per_index)
do
  python time_enclave.py models/mit.h5 0
done

for cut in {1..24}
do
  python build_enclave.py models/mit.h5 $cut
  
  for i in $(seq $runs_per_index)
  do
    python time_enclave.py models/mit_enclave.h5 $cut
    # echo $i
  done
done

cat timing_logs/mit_times.csv | mail -s "timing done" alexander.schloegl@uibk.ac.at

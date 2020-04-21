#!/bin/bash

exit_on_error() {
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        exit $exit_code
    fi
}


runs_per_index=20

if [ -z ${2+x} ];
then
    echo "Usage: $0 model_path num_cuts"
    exit 1;
else
    model_path=$1
    num_cuts=$2;
fi

basename=${model_path##*/}
dataset=${basename%.*}

make clean
make

# generate pure tf time
for i in $(seq $runs_per_index)
do
  python time_enclave.py ${model_path}.h5 0
  exit_on_error
done

for cut in $(seq $num_cuts)
do
  python build_enclave.py ${model_path}.h5 $cut
  exit_on_error
  
  for i in $(seq $runs_per_index)
  do
    python time_enclave.py ${model_path}_enclave.h5 $cut
    exit_on_error
    # echo $i
  done
done

cat "timing_logs/${dataset}_times.csv" | mail -s "timing done" alexander.schloegl@uibk.ac.at

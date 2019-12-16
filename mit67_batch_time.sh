#!/bin/bash

for batch_size in 7 8 9 10 15
do
    echo "Running batch size $batch_size"
    python mit67_time.py models/mit67.h5 models/mit67_enclave.h5 $batch_size >> timing_logs/mit67/23_layers/batch_size_$batch_size.log 2>> timing_logs/mit67/23_layers/batch_size_$batch_size.errlog

    echo -e "Batch size $batch_size is done.\nYay :)" | mail -s "Script execution finished" alexander.schloegl@uibk.ac.at
done

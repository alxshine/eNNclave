#!/bin/bash

num_layers=18
log_dir="timing_logs/mit67/${num_layers}_layers"
echo "logging outputs to ${log_dir}"

# create log directory if it doesn't exist
[ ! -d "${log_dir}" ] && mkdir -p ${log_dir}

for batch_size in 01 03 05 07 08 09 10 15
do
    echo "Running batch size ${batch_size}"
    output_log="${log_dir}/batch_size_${batch_size}.log"
    error_log="${log_dir}/batch_size_${batch_size}.errlog"
    python mit67_time.py models/mit67.h5 models/mit67_enclave.h5 ${batch_size} >> "${output_log}" 2>> "${error_log}"

    cat "${output_log}" | mail -s "MIT67 batch size ${batch_size} finished" alexander.schloegl@uibk.ac.at
done

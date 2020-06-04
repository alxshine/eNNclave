#!/bin/bash

docker build -t sgx-aesm --target aesm -f Dockerfile ..

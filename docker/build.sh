#!/bin/bash

docker build -t sgx-builder --target sgx-builder -f Dockerfile ..
docker build -t sgx-aesm --target sgx-aesm -f Dockerfile ..
docker build -t sgx-tester --target sgx-tester -f Dockerfile ..

rm -rf /tmp/aesmd
mkdir -p -m 777 /tmp/aesmd
chmod -R -f 777 /tmp/aesmd || sudo chmod -R -f 777 /tmp/aesmd || true
# docker-compose --verbose up
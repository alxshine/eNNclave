#!/bin/bash

sgx_dir=${HOME}/sgx
driver_dir=$sgx_dir/driver
sdk_dir=$sgx_dir/sdk

driver_git="https://github.com/intel/linux-sgx-driver"
sdk_git="https://github.com/intel/linux-sgx"

#general required tools
sudo apt-get install -y python3-pip build-essential git

mkdir -p $sgx_dir

#install linux-sgx-driver
sudo apt-get install -y linux-headers-$(uname -r)

git clone $driver_git $driver_dir
cd $driver_dir
make

sudo mkdir -p "/lib/modules/"`uname -r`"/kernel/drivers/intel/sgx"    
sudo cp isgx.ko "/lib/modules/"`uname -r`"/kernel/drivers/intel/sgx"    
sudo sh -c "cat /etc/modules | grep -Fxq isgx || echo isgx >> /etc/modules"    
sudo /sbin/depmod
sudo /sbin/modprobe isgx

#install SGX SDK and SGX PSW
sudo apt-get install -y ocaml ocamlbuild automake autoconf libtool wget python libssl-dev
sudo apt-get install libssl-dev libcurl4-openssl-dev protobuf-compiler libprotobuf-dev debhelper cmake

git clone $sdk_git $sdk_dir
cd $sdk_dir
./download_prebuilt.sh
make

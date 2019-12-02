#!/bin/bash

sgx_dir=${HOME}/sgx
driver_dir=$sgx_dir/installer/driver
sdk_dir=$sgx_dir/installer/sdk

driver_git="https://github.com/intel/linux-sgx-driver"
sdk_git="https://github.com/intel/linux-sgx"

#general required tools
sudo apt-get install -y python3-pip build-essential git

mkdir -p $sgx_dir

#install linux-sgx-driver
sudo apt-get install -y linux-headers-$(uname -r)

git clone --single-branch --branch sgx2 $driver_git $driver_dir
cd $driver_dir
make

sudo mkdir -p "/lib/modules/"`uname -r`"/kernel/drivers/intel/sgx"    
sudo cp isgx.ko "/lib/modules/"`uname -r`"/kernel/drivers/intel/sgx"    
sudo sh -c "cat /etc/modules | grep -Fxq isgx || echo isgx >> /etc/modules"    
sudo /sbin/depmod
sudo /sbin/modprobe isgx

#install SGX SDK and SGX PSW
sudo apt-get install -y ocaml ocamlbuild automake autoconf libtool wget python libssl-dev
sudo apt-get install -y libssl-dev libcurl4-openssl-dev protobuf-compiler libprotobuf-dev debhelper cmake

git clone $sdk_git $sdk_dir
cd $sdk_dir
./download_prebuilt.sh
make sdk_install_pkg

#install SGXSDK
$sdk_dir/linux/installer/bin/sgx_linux_x64_sdk_*.bin -prefix ${sgx_dir}
echo "source ${sgx_dir}/sgxsdk/environment" >> ${HOME}/.bashrc
source ${sgx_dir}/sgxsdk/environment

#install PSW
cd $sdk_dir
make deb_pkg
cd ${sdk_dir}/linux/installer/deb
sudo dpkg -i ./libsgx-urts_*.deb ./libsgx-enclave-common_*.deb ./libsgx-enclave-common-dev_*.deb

echo ""
echo ""
echo "SGX SUCCESSFULLY INSTALLED"
echo "You will need to install the python requirements listed in requirements.txt"
echo "Also, for the SGX SDK to work correctly, you need to run 'source ${sgx_dir}/sgxsdk/environment'"
echo "A corresponding line has been added to your .bashrc file, so this will be done automatically for future terminals"
echo ""
echo "For the code in this repository to work correctly, please run 'source ./setup_ld_path.sh'"
echo "Without this the program will not find the shared libraries generated during compilation"

#setup pip3
# pip3 install --user -r requirements.txt

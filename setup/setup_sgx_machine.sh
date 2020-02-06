#!/bin/bash

mkdir -p ~/Tools
sudo apt-get update

# install driver
cd ~/Tools/
sudo apt-get install linux-headers-$(uname -r)
git clone git@github.com:intel/linux-sgx-driver.git
cd linux-sgx-driver/
make
sh ./install.sh

# build SDK
cd ~/Tools/
sudo apt-get install build-essential ocaml ocamlbuild automake autoconf libtool wget python libssl-dev git
sudo apt-get install libssl-dev libcurl4-openssl-dev protobuf-compiler libprotobuf-dev debhelper cmake reprepro
git clone git@github.com:intel/linux-sgx.git
cd linux-sgx
./download_prebuilt.sh 
make -j8 sdk DEBUG=1
make -j8 sdk_install_pkg DEBUG=1

# install SDK
./linux/installer/bin/sgx* -prefix ~

# build PSW
make -j8 psw DEBUG=1
make -j8 deb_psw_package DEBUG=1
make -j8 deb_local_repo

# add local repo and install
sudo sh -c 'echo "deb [trusted=yes arch=amd64] file:$HOME/Tools/linux-sgx/linux/installer/deb/sgx_debian_local_repo bionic main" >> /etc/apt/sources.list'
suod apt-get update
sudo apt-get install libsgx-launch libsgx-launch-dbgsym libsgx-urts libsgx-urts-dbgsym libsgx-epid libsgx-epid-dbgsym libsgx-quote-ex libsgx-quote-ex-dbgsym

sudo systemctl enable aesmd
sudo systemctl start aesmd

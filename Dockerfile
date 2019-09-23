## -*- docker-image-name: "nn-sgx" -*-

FROM ubuntu:18.04

ENV driver="/linux-sgx-driver"

RUN apt-get update
RUN apt-get install -y git

###LINUX-SGX-DRIVER
#install prerequisites
RUN apt-get install -y linux-headers-generic build-essential

RUN echo "Linux SGX driver will be at ${driver}"
RUN git clone https://github.com/alxshine/linux-sgx-driver.git ${driver}
WORKDIR ${driver}

#build linux-sgx-driver
RUN make

#install linux-sgx-driver
# mkdir -p "/lib/modules/

# RUN apt-get install -y build-essential python3-dev

# WORKDIR /nn-sgx

# COPY . /nn-sgx

# RUN make

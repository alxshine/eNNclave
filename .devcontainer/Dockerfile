FROM ubuntu:18.04 as eNNclave-dev

RUN apt-get update && apt-get install -y \
    make cmake git g++ gcc

# SGX setup
WORKDIR /opt/intel
COPY --from=alxshine/linux-sgx:latest /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
RUN sh -c 'echo yes | ./sgx_linux_x64_sdk_*.bin'
ENV SGX_SDK=/opt/intel/sgxsdk
ENV PATH=${SGX_SDK}/bin:${SGX_SDK}/bin/x64:${PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SGX_SDK}/sdk_libs

# install conda (taken from conda/miniconda3 Dockerfile)
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

# build requisites
RUN apt-get update && apt-get install -y \
    build-essential python3.8-dev

RUN adduser --disabled-password --gecos '' ennclave
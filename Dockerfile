## -*- docker-image-name: "ennclave" -*-
###########################################################
#                Testing Environment                      #
###########################################################
FROM ubuntu:18.04 as eNNclave

RUN apt-get update && apt-get install -y \
  make cmake git g++ gcc

# SGX setup
WORKDIR /opt/intel
COPY --from=alxshine/linux-sgx:latest /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
RUN sh -c 'echo yes | ./sgx_linux_x64_sdk_*.bin'

# build requisites
RUN apt-get update && apt-get install -y \
  build-essential python3.8-dev

WORKDIR /eNNclave

COPY environment.yml /eNNclave/

COPY setup_ld_path.sh /eNNclave/
COPY frontend /eNNclave/frontend
COPY backend /eNNclave/backend
COPY core /eNNclave/core
COPY inc /eNNclave/inc
COPY CMakeLists.txt googletest_CMakeLists.txt /eNNclave/

ENV ENNCLAVE_HOME /eNNclave
ENV LD_LIBRARY_PATH=/eNNclave/lib:$LD_LIBRARY_PATH
ENV CXX g++
ENV CC gcc
RUN mkdir /eNNclave/lib && mkdir -p /eNNclave/backend/generated && mkdir /eNNclave/build


CMD ["bash"]

FROM eNNclave as eNNclave-tester

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

RUN conda env create

COPY test_docker.sh /eNNclave

# CMD "bash"
CMD ["bash", "test_docker.sh"]

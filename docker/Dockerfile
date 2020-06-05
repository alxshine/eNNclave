###########################################################
#                       SGX Install                       #
###########################################################
FROM ubuntu:18.04 as sgx-builder

RUN apt-get update && apt-get install -y \
  autoconf automake build-essential cmake git \
  curl wget debhelper \
  libcurl4-openssl-dev libprotobuf-dev libssl-dev libtool \
  lsb-release ocaml ocamlbuild protobuf-compiler python

ADD linux-sgx /linux-sgx
WORKDIR /linux-sgx

RUN ./download_prebuilt.sh
RUN make sdk_install_pkg_no_mitigation

WORKDIR /opt/intel
RUN sh -c 'echo yes | /linux-sgx/linux/installer/bin/sgx_linux_x64_sdk_*.bin'

WORKDIR /linux-sgx
RUN make psw_install_pkg

###########################################################
#                    AESM service                         #
###########################################################
FROM ubuntu:18.04 as sgx-aesm
RUN apt-get update && apt-get install -y \
  libcurl4 libprotobuf10 libssl1.1 make module-init-tools

WORKDIR /installer
COPY --from=sgx-builder /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
USER aesmd
WORKDIR /opt/intel/sgxpsw/aesm
ENV LD_LIBRARY_PATH=.
CMD ./aesm_service && bash

###########################################################
#                Testing Environment                      #
###########################################################
FROM ubuntu:18.04 as sgx-tester

RUN apt-get update && apt-get install -y \
  libcurl4-openssl-dev libprotobuf-dev libssl-dev \
  g++ make module-init-tools

# SGX setup
WORKDIR /opt/intel
COPY --from=sgx-builder /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
RUN sh -c 'echo yes | ./sgx_linux_x64_sdk_*.bin'

RUN \
  adduser --disabled-password --gecos "" ennclave

COPY --chown=ennclave:ennclave lib /eNNclave/lib
COPY --chown=ennclave:ennclave interop /eNNclave/interop
COPY --chown=ennclave:ennclave inc /eNNclave/inc
COPY --chown=ennclave:ennclave Makefile *.py *.sh /eNNclave/
COPY --chown=ennclave:ennclave setup/setup_ld_path.sh docker/tester_entrypoint.sh /eNNclave/setup/

USER ennclave
WORKDIR /eNNclave


RUN make clean

ENV LD_LIBRARY_PATH="/eNNclave/lib:$LD_LIBRARY_PATH"


# RUN pip3 install \
#   tensorflow pandas numpy

ENTRYPOINT [ "/eNNclave/setup/tester_entrypoint.sh" ]
CMD ["bash"]

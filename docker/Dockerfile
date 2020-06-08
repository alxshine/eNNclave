###########################################################
#                Testing Environment                      #
###########################################################
FROM ubuntu:18.04 as sgx-tester

RUN apt-get update && apt-get install -y \
  make

# SGX setup
WORKDIR /opt/intel
COPY --from=sgx_builder /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
RUN sh -c 'echo yes | ./sgx_linux_x64_sdk_*.bin'

# build requisites
RUN apt-get install -y \
  build-essential python3.7-dev

WORKDIR /eNNclave
RUN \
  adduser --disabled-password --gecos "" ennclave && chown ennclave:ennclave -R /eNNclave

COPY --chown=ennclave:ennclave lib /eNNclave/lib
COPY --chown=ennclave:ennclave interop /eNNclave/interop
COPY --chown=ennclave:ennclave inc /eNNclave/inc
COPY --chown=ennclave:ennclave Makefile *.py *.sh /eNNclave/
COPY --chown=ennclave:ennclave docker/init.sh setup_ld_path.sh /eNNclave/

USER ennclave

RUN make clean

COPY --chown=ennclave:ennclave parameters.bin /eNNclave/

# RUN pip3 install \
#   tensorflow pandas numpy

CMD ["bash"]

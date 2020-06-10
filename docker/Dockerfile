###########################################################
#                Testing Environment                      #
###########################################################
FROM ubuntu:18.04 as sgx-tester

RUN apt-get update && apt-get install -y \
  make

# SGX setup
WORKDIR /opt/intel
COPY --from=alxshine/linux-sgx:latest /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
RUN sh -c 'echo yes | ./sgx_linux_x64_sdk_*.bin'

# build requisites
RUN apt-get install -y \
  build-essential python3.7-dev python3-pip

WORKDIR /eNNclave
RUN \
  adduser --disabled-password --gecos "" ennclave && chown ennclave:ennclave -R /eNNclave

COPY --chown=ennclave:ennclave requirements_cpu.txt /eNNclave/requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

COPY --chown=ennclave:ennclave docker/run.sh setup_ld_path.sh /eNNclave/
COPY --chown=ennclave:ennclave inc /eNNclave/inc
COPY --chown=ennclave:ennclave lib /eNNclave/lib
COPY --chown=ennclave:ennclave interop /eNNclave/interop
COPY --chown=ennclave:ennclave Makefile *.py *.sh /eNNclave/


USER ennclave
ENV MODE=SIM

RUN make clean

COPY --chown=ennclave:ennclave models /eNNclave/models

# COPY --chown=ennclave:ennclave parameters.bin /eNNclave/

# RUN pip3 install \
#   tensorflow pandas numpy

CMD bash

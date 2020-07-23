## -*- docker-image-name: "ennclave" -*-
###########################################################
#                Testing Environment                      #
###########################################################
FROM ubuntu:18.04 as sgx-base

RUN apt-get update && apt-get install -y \
  make

# SGX setup
WORKDIR /opt/intel
COPY --from=alxshine/linux-sgx:latest /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
RUN sh -c 'echo yes | ./sgx_linux_x64_sdk_*.bin'

# build requisites
RUN apt-get update && apt-get install -y \
  build-essential python3.7-dev python3-pip

WORKDIR /eNNclave
RUN \
  adduser --disabled-password --gecos "" ennclave && chown ennclave:ennclave -R /eNNclave

COPY --chown=ennclave:ennclave requirements.txt /eNNclave/requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

USER ennclave

COPY --chown=ennclave:ennclave setup_ld_path.sh /eNNclave/
COPY --chown=ennclave:ennclave inc /eNNclave/inc
COPY --chown=ennclave:ennclave lib /eNNclave/lib
COPY --chown=ennclave:ennclave interop /eNNclave/interop
COPY --chown=ennclave:ennclave Makefile *.py *.sh /eNNclave/

ENV MODE=SIM

RUN make clean
RUN mkdir timing_logs

FROM sgx-base AS tester

COPY --chown=ennclave:ennclave tests/run_tests.sh /eNNclave/
COPY --chown=ennclave:ennclave tests/*.py /eNNclave/tests/

# CMD ["bash"]
CMD ["bash", "./run_tests.sh"]

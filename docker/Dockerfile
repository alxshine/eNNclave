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
RUN apt-get update && apt-get install -y \
  build-essential python3.7-dev python3-pip

WORKDIR /eNNclave
RUN \
  adduser --disabled-password --gecos "" ennclave && chown ennclave:ennclave -R /eNNclave

COPY --chown=ennclave:ennclave requirements_cpu.txt /eNNclave/requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

USER ennclave
RUN mkdir -p datasets/mnist
COPY --chown=ennclave:ennclave datasets/mnist datasets/mnist

COPY --chown=ennclave:ennclave docker/run_unit_tests.sh docker/run_integration_tests.sh setup_ld_path.sh /eNNclave/
COPY --chown=ennclave:ennclave inc /eNNclave/inc
COPY --chown=ennclave:ennclave lib /eNNclave/lib
COPY --chown=ennclave:ennclave interop /eNNclave/interop
COPY --chown=ennclave:ennclave Makefile *.py *.sh /eNNclave/

ENV MODE=SIM

RUN make clean
RUN mkdir timing_logs

COPY --chown=ennclave:ennclave models /eNNclave/models

CMD ["bash", "./run_integration_tests.sh"]
# CMD ["bash"]
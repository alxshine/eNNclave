FROM ubuntu:18.04 as build-sgx

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

FROM ubuntu:18.04 as aesm
RUN apt-get update && apt-get install -y \
  libcurl4 libprotobuf10 libssl1.1 make module-init-tools

WORKDIR /installer
COPY --from=build-sgx /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
USER aesmd
WORKDIR /opt/intel/sgxpsw/aesm
ENV LD_LIBRARY_PATH=.
CMD ./aesm_service --no-daemon

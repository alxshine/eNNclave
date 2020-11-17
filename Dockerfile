## -*- docker-image-name: "ennclave" -*-

###########################################################
#                     Base Environment                    #
###########################################################
FROM ubuntu:18.04 as eNNclave

RUN apt-get update && apt-get install -y \
  make cmake git g++ gcc

# SGX setup
WORKDIR /opt/intel
COPY --from=alxshine/linux-sgx:latest /linux-sgx/linux/installer/bin/*.bin ./
RUN ./sgx_linux_x64_psw*.bin --no-start-aesm
RUN sh -c 'echo yes | ./sgx_linux_x64_sdk_*.bin'

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

WORKDIR /eNNclave

COPY environment.yml /eNNclave/
RUN conda env create

CMD ["bash"]

###########################################################
#              Development Environment                    #
###########################################################
FROM eNNclave as eNNclave-dev

RUN apt-get update -y && \
  apt-get install -y --no-install-recommends wget openssh-server mosh

RUN mkdir /var/run/sshd \
  && echo 'AuthorizedKeysFile %h/.ssh/authorized_keys' >> /etc/ssh/sshd_config \
  && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd \
  && sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
  && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

ARG github_users
RUN test -n "$github_users"

ARG SSH=/root/.ssh

RUN mkdir ${SSH}
RUN chmod 700 ${SSH}
RUN wget "https://github.com/${github_users}.keys" -O ${SSH}/authorized_keys
RUN chmod 600 ${SSH}/authorized_keys

COPY docker.bashrc /root/.bashrc

CMD ["/usr/sbin/sshd", "-D"]

###########################################################
#                Testing Environment                      #
###########################################################
FROM eNNclave as eNNclave-tester

COPY setup_ld_path.sh /eNNclave/
COPY frontend /eNNclave/frontend
COPY backend /eNNclave/backend
COPY core /eNNclave/core
COPY inc /eNNclave/inc
COPY CMakeLists.txt googletest_CMakeLists.txt /eNNclave/

RUN mkdir /eNNclave/lib && mkdir -p /eNNclave/backend/generated && mkdir /eNNclave/build
RUN (cd /eNNclave/build && cmake ..)

COPY test_docker.sh /eNNclave

# CMD "bash"
CMD ["bash", "test_docker.sh"]

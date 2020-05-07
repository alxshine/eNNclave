ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

ifeq ($(SGX_MODE), SIM)
USER_LIBS:=-lnative
else
USER_LIBS:=-lnative -lenclave
endif

HOST=$(shell [ -d /usr/include/python3.7m ] && echo "-DHOST")

all: pymatutilmodule.c
	$(CC) -fPIC -g -shared pymatutilmodule.c -I${INC} -L${LIB} $(HOST) $(USER_LIBS) -o pymatutil.so

.PHONY:
clean:
	rm -rf pymatutil.so

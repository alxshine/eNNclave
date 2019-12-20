ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

HOST=$(shell [ -d /usr/include/python3.7m ] && echo "-DHOST")

all: pymatutilmodule.c
	$(CC) -fPIC -g -shared pymatutilmodule.c -I${INC} -L${LIB} $(HOST) -lenclave -lnative -o pymatutil.so

.PHONY:
clean:
	rm -rf pymatutil.so

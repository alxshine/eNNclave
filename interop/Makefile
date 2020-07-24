ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

DEFINES=
ifeq ($(SGX_SDK),)
USER_LIBS:=-lnative
else
USER_LIBS:=-lnative -lenclave
DEFINES+=-DSGX_SDK
endif

PYTHON_INC=/usr/include/python3.8

all: pymatutilmodule.c
	$(CC) -fPIC -g -shared pymatutilmodule.c -I${INC} -I${PYTHON_INC} -L${LIB} $(USER_LIBS) ${DEFINES} -o pymatutil.so

.PHONY:
clean:
	rm -rf pymatutil.so

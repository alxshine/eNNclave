ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

all: pymatutilmodule.c
	$(CC) -fPIC -g -shared pymatutilmodule.c -I${INC} -L${LIB} -lmatutil_native -o pymatutil.so

.PHONY:
clean:
	rm -rf pymatutil.so

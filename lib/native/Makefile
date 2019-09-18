ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

all: matutil.c
	clang -g -shared -I${INC} matutil.c -o libmatutil_native.so
	mv libmatutil_native.so ..

.PHONY:
clean:
	rm -rf *.so

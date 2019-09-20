ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

all: matutil.c
	$(CC) -g -shared -fPIC -I${INC} matutil.c state.c -o libmatutil.so
	cp libmatutil.so ..

.PHONY:
clean:
	rm -rf *.so

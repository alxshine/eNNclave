ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

all: matutil.cpp dense.cpp state.cpp
	$(CC) -g -shared -fPIC -I${INC} $^ -o libmatutil.so
	cp libmatutil.so ..

.PHONY:
clean:
	rm -rf *.so

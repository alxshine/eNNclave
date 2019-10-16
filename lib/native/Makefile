ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

weight_binaries:= $(wildcard *.bin)
weight_objects:=$(patsubst %.bin,%.o,$(weight_binaries))

all: matutil.cpp dense.cpp state.cpp $(weight_objects)
	$(CC) -g -shared -fPIC -I${INC} $^ -o libmatutil.so
	cp libmatutil.so ..

%.o: %.bin
	ld -r -b binary $< -o $@

.PHONY:
clean:
	rm -rf *.so *.o

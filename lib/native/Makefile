ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

weight_binaries:= $(wildcard *.bin)
weight_objects:=$(patsubst %.bin,%.o,$(weight_binaries))

all: matutil.cpp forward.cpp state.cpp native_nn.cpp $(weight_objects)
	$(CC) -g -shared -fPIC -I${INC} $^ -o libnative.so
	ln -sf native/libnative.so ../

%.o: %.bin
	ld -r -b binary $< -o $@

.PHONY:
clean:
	rm -rf *.so *.o *.bin forward.c state.c
	cp ../dummy_forward.c forward.c
	cp ../dummy_state.c state.c

ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

weight_binaries:= $(wildcard *.bin)
weight_objects:=$(patsubst %.bin,%.o,$(weight_binaries))

all: matutil.cpp forward.cpp state.cpp $(weight_objects)
	$(CC) -g -shared -fPIC -I${INC} $^ -o libmatutil.so
	ln -sf native/libmatutil.so ../

%.o: %.bin
	ld -r -b binary $< -o $@

.PHONY:
clean:
	rm -rf *.so *.o *.bin forward.cpp state.cpp
	cp ../dummy_forward.cpp forward.cpp
	cp ../dummy_state.cpp state.cpp

ROOT=../..
INC=${ROOT}/inc
LIB=${ROOT}/lib

weight_binaries:= $(wildcard *.bin)
weight_objects:=$(patsubst %.bin,%.o,$(weight_binaries))

all: ../common/matutil.c forward.c native_nn.c parameters.c output.c
	$(CC) -g -shared -fPIC -I${INC} $^ -o libnative.so
	ln -sf native/libnative.so ../

%.o: %.bin
	ld -r -b binary $< -o $@

.PHONY:
clean:
	rm -rf *.so *.o *.bin forward.c state.c
	cp ../native_dummy_forward.c forward.c


export ROOT:=$(shell pwd)
export INC:=${ROOT}/inc
export LIB:=${ROOT}/lib

all: lib interop

lib:
	$(MAKE) -C lib -e
interop:
	$(MAKE) -C interop -e

test:
	$(MAKE) -C lib -e test

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C interop clean
	rm -f enclave.so enclave.signed.so

.PHONY: lib python interop clean

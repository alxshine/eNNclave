export ROOT:=$(shell pwd)
export INC:=${ROOT}/inc
export LIB:=${ROOT}/lib

all: lib interop
lib:
	$(MAKE) -C lib -e
interop:
	$(MAKE) -C interop -e

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C interop clean

.PHONY: lib interop clean

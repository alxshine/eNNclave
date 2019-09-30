export ROOT:=$(shell pwd)
export INC:=${ROOT}/inc
export LIB:=${ROOT}/lib

all:
	$(MAKE) lib
	$(MAKE) python

lib:
	$(MAKE) -C lib -e
python:
	$(MAKE) -C python -e

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C python clean

.PHONY: lib python clean

SUBDIRS := lib interop
SUBDIRSCLEAN=$(addsuffix clean,$(SUBDIRS))

export ROOT:=$(shell pwd)
export INC:=${ROOT}/inc
export LIB:=${ROOT}/lib

all:
	$(MAKE) -C lib -e
	$(MAKE) -C interop -e

clean:
	$(MAKE) -C lib clean
	$(MAKE) -C interop clean

.PHONY: clean

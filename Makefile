SUBDIRS := interop lib

export ROOT:=$(shell pwd)
export INC:=${ROOT}/inc
export LIB:=${ROOT}/lib

all: $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@ -e

SUBDIRSCLEAN=$(addsuffix clean,$(SUBDIRS))
clean: $(SUBDIRSCLEAN)

%clean: %
	$(MAKE) -C $< clean

.PHONY: $(SUBDIRS)

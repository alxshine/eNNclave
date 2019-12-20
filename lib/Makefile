ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

all:
	@printf "\n\nMaking library in native mode\n\n\n"
	@$(MAKE) -C native
	@printf "\n\nMaking library in sgx mode\n\n\n"
	@$(MAKE) -C enclave
	@ln -sf libenclave.so libmatutil.so

.PHONY: clean
clean:
	rm -rf *.o test_matutil *.so
	make -C native clean
	make -C enclave clean

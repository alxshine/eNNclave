ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

all: native enclave

native:
	@printf "\n\nMaking library in native mode\n\n\n"
	@$(MAKE) -C native

enclave:
	@printf "\n\nMaking library in sgx mode\n\n\n"
	@$(MAKE) -C enclave

.PHONY: clean native enclave
clean:
	rm -rf *.o test_matutil *.so
	make -C native clean
	make -C enclave clean

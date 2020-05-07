ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

ifeq ($(SGX_MODE), SIM)

TEST_LIB=-lnative
libs: native
	@printf "\n\n****** NOT BUILDING SGX LIBRARY ******\n\n\n"

else

TEST_LIB=-lenclave
libs: native enclave

endif

all: libs

native:
	@printf "\n\nMaking library in native mode\n\n\n"
	@$(MAKE) -C native

enclave:
	@printf "\n\nMaking library in sgx mode\n\n\n"
	@$(MAKE) -C enclave
	@ln -sf lib/enclave.so ../
	@ln -sf lib/enclave.signed.so ../

test: test_matutil.c libs
	gcc -g $< -I$(INC) -L$(LIB) -L$(CURDIR) $(TEST_LIB) -o test

.PHONY: clean native enclave
clean:
	rm -rf *.o test_matutil *.so test
	make -C native clean
	make -C enclave clean

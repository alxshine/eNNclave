ROOT=..
INC=${ROOT}/inc
LIB=${ROOT}/lib

ifeq ($(TEST_MODE), native)
TEST_LIB=-lnative
test_lib: native
	@printf "\n\n****** TESTING NATIVE ******\n\n\n"
else
TEST_LIB=-lenclave
test_lib: enclave
	@printf "\n\n****** TESTING ENCLAVE ******\n\n\n"
endif

all: native enclave

native:
	@printf "\n\nMaking library in native mode\n\n\n"
	@$(MAKE) -C native

enclave:
	@printf "\n\nMaking library in sgx mode\n\n\n"
	@$(MAKE) -C enclave
	@ln -sf lib/enclave.so ../
	@ln -sf lib/enclave.signed.so ../

test: test_matutil.c native
	gcc -g $< -I$(INC) -L$(LIB) -lnative -o test_matutil

.PHONY: clean native enclave
clean:
	rm -rf *.o test_matutil *.so test
	make -C native clean
	make -C enclave clean

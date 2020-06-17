ROOT=..
INC=${ROOT}/inc
LIB=${ROOT}/lib

TEST_FILES=$(wildcard test/*.c)

all: native enclave

native:
	@printf "\n\nMaking library in native mode\n\n\n"
	@$(MAKE) -C native

enclave:
	@printf "\n\nMaking library in sgx mode\n\n\n"
	@$(MAKE) -C enclave
	@ln -sf lib/enclave.so ../
	@ln -sf lib/enclave.signed.so ../

test: test_matutil.c $(TEST_FILES) native
	gcc -g $< $(TEST_FILES) -I$(INC) -L$(LIB) -lnative -o test_matutil

.PHONY: clean native enclave
clean:
	rm -f *.o test_matutil *.so test_matutil
	make -C native clean
	make -C enclave clean

ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

all: native sgx test_native

native:
	$(MAKE) -C native

sgx:
	$(MAKE) -C sgx
	cp sgx/enclave.so sgx/libmatutil_sgx.so .

test_native: test_matutil.c
	$(CC) -g test_matutil.c -lmatutil_native -I${INC} -L${LIB} -o test_native

.PHONY: native sgx
clean:
	rm -rf *.o test_native *.so
	make -C native clean
	make -C sgx clean

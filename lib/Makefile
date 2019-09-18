ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

all: test_native

test_native: test_matutil.c
	clang -g test_matutil.c -lmatutil_native -I${INC} -L${LIB} -o test_native

.PHONY:
clean:
	rm -rf *.o test_native

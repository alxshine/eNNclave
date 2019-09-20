ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib
MODE:=native

all:
	$(MAKE) lib
	$(MAKE) test

lib:
	$(MAKE) -C $(MODE)

test: test_matutil.c
	$(CC) -g test_matutil.c -lmatutil -I${INC} -L${LIB} -o test_matutil

.PHONY: native sgx
clean:
	rm -rf *.o test_matutil *.so
	make -C native clean
	make -C sgx clean

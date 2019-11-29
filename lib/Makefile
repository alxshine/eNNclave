ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib
MODE:=native

all:
	@$(MAKE) lib
ifeq ($(MODE), native)
	@$(MAKE) test
endif

lib:
	@printf "\n\nMaking library in $(MODE) mode\n\n\n"
	$(MAKE) -C $(MODE)

test: test_matutil.c
	$(CC) -g test_matutil.c -lmatutil -I${INC} -L${LIB} -o test_matutil

.PHONY: native sgx
clean:
	rm -rf *.o test_matutil *.so
	make -C native clean
	make -C sgx clean

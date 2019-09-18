all: test_native

test_native: test_matutil.c
	clang -g test_matutil.c -lmatutil -Lnative -o test_native

.PHONY:
clean:
	rm -rf *.o test_native

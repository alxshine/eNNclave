all: matutil.c
	clang -g -shared -I.. matutil.c -o libmatutil_native.so
	mv libmatutil_native.so ..

.PHONY:
clean:
	rm -rf *.so

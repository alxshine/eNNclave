all: matutil.c
	clang -g -shared -I.. matutil.c -o libmatutil.so

.PHONY:
clean:
	rm -rf *.so

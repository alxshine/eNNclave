all: pymatutilmodule.c
	clang -fPIC -g -shared pymatutilmodule.c -I../matutil -Lnative -lmatutil -o pymatutil.so

.PHONY:
clean:
	rm -rf pymatutil.so

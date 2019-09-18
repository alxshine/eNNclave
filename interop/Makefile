all: pymatutilmodule.c
	clang -fPIC -g -shared pymatutilmodule.c -Lnative -lmatutil -o pymatutil.so

.PHONY:
clean:
	rm -rf pymatutil.so

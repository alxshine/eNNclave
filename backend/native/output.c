#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "output.h"

void print_err(const char *fmt, ...) {
  char buf[BUFSIZ] = {'\0'};
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, BUFSIZ, fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s", buf);
}

void print_out(const char *fmt, ...) {
  char buf[BUFSIZ] = {'\0'};
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, BUFSIZ, fmt, ap);
  va_end(ap);
  fprintf(stdout, "%s", buf);
}
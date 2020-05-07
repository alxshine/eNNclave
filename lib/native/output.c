#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "output.h"

int print_err(const char *fmt, ...) {
  char buf[BUFSIZ] = {'\0'};
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, BUFSIZ, fmt, ap);
  va_end(ap);
  fprintf(stderr, "%s", buf);
  return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

int print_out(const char *fmt, ...) {
  char buf[BUFSIZ] = {'\0'};
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, BUFSIZ, fmt, ap);
  va_end(ap);
  fprintf(stdout, "%s", buf);
  return (int)strnlen(buf, BUFSIZ - 1) + 1;
}
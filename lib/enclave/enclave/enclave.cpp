#include "enclave.h"
#include "enclave_t.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

void test(){
    print("This is the enclave :)\n");
}

/*  
 *  * printf:  
 *  *   Invokes OCALL to display the enclave buffer to the terminal. 
 *  */
int print(const char* fmt, ...)
{
  char buf[BUFSIZ] = { '\0' };
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, BUFSIZ, fmt, ap);
  va_end(ap);
  ocall_stdout_string(buf);
  return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

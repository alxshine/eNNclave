######## Project Variables ########

export ROOT:=${HOME}/Projects/nn-sgx
export INC:=${ROOT}/inc
export LIB:=${ROOT}/lib

######## Actual Make ########

wrapper_name:= libenclave.so
enclave_name:= enclave.so
signed_enclave_name= enclave.signed.so

SGX_LIBRARY_PATH=$(SGX_SDK)/lib64
TRTS=sgx_trts
CRYPTO=sgx_tcrypto
SERVICE=sgx_tservice
STDLIB=sgx_tstdc
STDCXX=sgx_tcxx

enclave_link_flags := $(wrapper_link_flags) \
-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_LIBRARY_PATH) \
-Wl,--whole-archive -l$(TRTS) -l$(CRYPTO) -l$(SERVICE) -Wl,--no-whole-archive \
-Wl,--start-group -l$(STDLIB) -l$(STDCXX) -Wl,--end-group \
-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
-Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections   \
-Wl,--version-script=enclave/enclave.lds
c_flags = -fPIC -g
enclave_includes =-I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx -I$(INC)
wrapper_includes =-I$(SGX_SDK)/include -I$(INC)
link_flags =-L$(SGX_LIBRARY_PATH) -lsgx_urts -lpthread -l:libsgx_capable.a

all: $(wrapper_name) $(signed_enclave_name)

# generate code files
edger8r: enclave/enclave.edl
	@sgx_edger8r --trusted --untrusted --trusted-dir enclave/ --untrusted-dir wrapper/ --search-path enclave/ enclave/enclave.edl

# App stuff

$(wrapper_name): wrapper/enclave_u.o wrapper/wrapper.o wrapper/utils.o
	$(CXX) $(wrapper_includes) -shared $(c_flags) $^ -o $@ $(link_flags)

wrapper/enclave_u.h: edger8r

wrapper/enclave_u.c: wrapper/enclave_u.h

wrapper/enclave_u.o: wrapper/enclave_u.c
	@$(CC) $(enclave_includes) $(c_flags) $^ -c -o $@ $(link_flags)
	
wrapper/%.o: wrapper/%.cpp
	@$(CXX) $(wrapper_includes) $(c_flags) $^ -c -o $@

# Enclave stuff
enclave/enclave_t.h: edger8r

enclave/enclave_t.o: enclave/enclave_t.c
	@$(CC) $(enclave_includes) -fPIC $^ -c -o $@ $(enclave_link_flags)

$(enclave_name): enclave/enclave_t.o enclave/enclave.o enclave/forward.o enclave/matutil.o
	$(CXX) $(enclave_includes) $^ -o $@ $(enclave_link_flags)

enclave/%.o: enclave/%.cpp
	$(CXX) $(enclave_includes) -fPIC $^ -c -o $@ $(enclave_link_flags)

$(signed_enclave_name): $(enclave_name)
	sgx_sign sign -enclave $< -out $@ -key enclave/signing_key.pem -config enclave/config.xml

.PHONY: clean
clean:
	rm -rf *.so testitest */*.o wrapper/*_u.* enclave/*_t.*

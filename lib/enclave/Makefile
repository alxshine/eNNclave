ROOT=${HOME}/Projects/nn-sgx
INC=${ROOT}/inc
LIB=${ROOT}/lib

app_name:= testitest
enclave_name:= enclave.so
signed_enclave_name= enclave.signed.so
wrapper_name:= libenclave.so

C_FLAGS = -fPIC -g -DSGX_DEBUG_FLAG=1

SGX_LIBRARY_PATH=$(SGX_SDK)/lib64
TRTS=sgx_trts
CRYPTO=sgx_tcrypto
SERVICE=sgx_tservice
STDLIB=sgx_tstdc
STDCXX=sgx_tcxx

enclave_link_flags := $(app_link_flags) \
-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_LIBRARY_PATH) \
-Wl,--whole-archive -l$(TRTS) -l$(CRYPTO) -l$(SERVICE) -Wl,--no-whole-archive \
-Wl,--start-group -l$(STDLIB) -l$(STDCXX) -Wl,--end-group \
-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
-Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections   \
-Wl,--version-script=enclave/enclave.lds

enclave_includes =-I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx -I$(INC)
app_includes =-I$(SGX_SDK)/include -I$(INC)
link_flags =-L$(SGX_LIBRARY_PATH) -lsgx_urts -lpthread -l:libsgx_capable.a

all: $(wrapper_name) $(app_name) $(signed_enclave_name)

# generate code files
edger8r: enclave/enclave.edl
	@sgx_edger8r --trusted --untrusted --trusted-dir enclave/ --untrusted-dir app/ --search-path enclave/ enclave/enclave.edl

# Wrapper stuf
$(wrapper_name): app/enclave_u.o app/wrapper.o app/utils.o
	$(CXX) $(app_includes) $(C_FLAGS) -shared $^ -o $@ $(link_flags)
	ln -sf enclave/$@ ../

# App stuff
$(app_name): app/enclave_u.o app/app.o app/utils.o
	$(CXX) $(app_includes) $(C_FLAGS) $^ -o $@ $(link_flags) -L$(CURDIR) -lenclave

app/enclave_u.h: edger8r

app/enclave_u.c: app/enclave_u.h

app/enclave_u.o: app/enclave_u.c
	$(CC) $(enclave_includes) $(C_FLAGS) $^ -c -o $@ $(link_flags)
	
app/%.o: app/%.cpp
	$(CXX) $(app_includes) $(C_FLAGS) $^ -c -o $@

# Enclave stuff
enclave/enclave_t.h: edger8r

enclave/enclave_t.c: enclave/enclave_t.h

enclave/enclave_t.o: enclave/enclave_t.c
	$(CC) $(enclave_includes) $(C_FLAGS) $^ -c -o $@ $(enclave_link_flags)

$(signed_enclave_name): $(enclave_name)
	sgx_sign sign -enclave $< -out $@ -key enclave/signing_key.pem -config enclave/config.xml

$(enclave_name): enclave/enclave_t.o enclave/enclave.o enclave/matutil.o
	$(CXX) $(enclave_includes) $^ -o $@ $(enclave_link_flags)

enclave/%.o: enclave/%.cpp
	$(CXX) $(enclave_includes) $^ -c -o $@ $(enclave_link_flags)
	
.PHONY: clean
clean:
	rm -rf *.so testitest */*.o app/*_u.* enclave/*_t.*
	cp ../dummy_forward.cpp enclave/forward.cpp
	cp ../dummy_state.cpp enclave/state.cpp

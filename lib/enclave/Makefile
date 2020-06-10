ROOT=../..
INC=${ROOT}/inc
LIB=${ROOT}/lib

app_name:= encrypt_parameters
enclave_name:= enclave.so
signed_enclave_name= enclave.signed.so
wrapper_name:= libenclave.so

C_FLAGS = -fPIC -g -DSGX_DEBUG_FLAG=1

SGX_LIBRARY_PATH=$(SGX_SDK)/lib64
ifeq ($(MODE), SIM)
	URTS=sgx_urts_sim
	TRTS=sgx_trts_sim
	MESSAGE:="Built enclave in SIMULATION mode"
else
	URTS=sgx_urts
	TRTS=sgx_trts
	MESSAGE:="Built enclave in HARDWARE mode"
endif

CRYPTO=sgx_tcrypto
SERVICE=sgx_tservice
STDLIB=sgx_tstdc
TCMALLOC=sgx_tcmalloc
STDCXX=sgx_tcxx

enclave_link_flags := $(app_link_flags) \
-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_LIBRARY_PATH) \
-Wl,--whole-archive \
-l$(TRTS) \
-l$(CRYPTO) \
-l$(SERVICE) \
-lsgx_tprotected_fs \
-Wl,--no-whole-archive \
-Wl,--start-group \
-l$(STDLIB) \
-l$(STDCXX) \
-Wl,--end-group \
-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
-Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections   \
-Wl,--version-script=enclave/enclave.lds

enclave_includes =-I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx -I$(INC)
app_includes =-I$(SGX_SDK)/include -I$(INC)
link_flags =-L$(SGX_LIBRARY_PATH) -l$(URTS) -lpthread -l:libsgx_capable.a -lsgx_uprotected_fs

binary_output_files = $(patsubst %.bin,%_bin.o,$(wildcard enclave/*.bin))

all: $(wrapper_name) $(app_name) $(signed_enclave_name) parameters
	@printf "\n\n%s\n\n\n" $(MESSAGE)

# generate code files
edger8r: enclave/enclave.edl
	@sgx_edger8r --trusted --untrusted --trusted-dir enclave/ --untrusted-dir app/ --search-path enclave/ enclave/enclave.edl --search-path $(SGX_SDK)/include/

# Wrapper stuf
$(wrapper_name): app/enclave_u.o app/wrapper.o app/utils.o
	$(CC) $(app_includes) $(C_FLAGS) -shared $^ -o $@ $(link_flags)
	ln -sf enclave/$@ ../

# App stuff
$(app_name): app/enclave_u.o app/app.o app/utils.o
	$(CC) $(app_includes) $(C_FLAGS) $^ -o $@ $(link_flags) -L$(CURDIR) -lenclave

app/enclave_u.h: edger8r

app/enclave_u.c: app/enclave_u.h

app/enclave_u.o: app/enclave_u.c
	$(CC) $(enclave_includes) $(C_FLAGS) $^ -c -o $@ $(link_flags)
	
app/%.o: app/%.c
	$(CC) $(app_includes) $(C_FLAGS) $^ -c -o $@

# Enclave stuff
enclave/enclave_t.h: edger8r

enclave/enclave_t.c: enclave/enclave_t.h

enclave/enclave_t.o: enclave/enclave_t.c
	$(CC) $(enclave_includes) $(C_FLAGS) $^ -c -o $@ $(enclave_link_flags)

$(signed_enclave_name): $(enclave_name)
	sgx_sign sign -enclave $< -out $@ -key enclave/signing_key.pem -config enclave/config.xml
	@ln -sf enclave/enclave.signed.so ../

$(enclave_name): enclave/enclave_t.o enclave/enclave.o ../common/matutil.c enclave/forward.o enclave/parameters.o
	$(CC) $(enclave_includes) $^ -o $@ $(enclave_link_flags)
	@ln -sf enclave/enclave.so ../

enclave/%_bin.o: enclave/%.bin
	@echo $^
	@(cd enclave && ld -r -b binary $(notdir $<) -o $(notdir $@))

enclave/%.o: enclave/%.c
	$(CC) $(enclave_includes) $(C_FLAGS) $^ -c -o $@ $(enclave_link_flags)

# generate parameters
parameters: $(app_name) $(enclave_name) $(wrapper_name)
	@echo "Encrypting parameters"
	@rm -f ../../encrypted_params.aes.bin
	@./$(app_name) && mv ./encrypted_params.aes.bin ../../
	
.PHONY: clean
clean:
	rm -rf *.so testitest */*.o app/*_u.* enclave/*_t.* enclave/forward.c* enclave/state* enclave/*.bin
	cp ../enclave_dummy_forward.c enclave/forward.c
	cp ../enclave_dummy_config.xml enclave/config.xml
	rm -f ../../encrypted_params.aes.bin

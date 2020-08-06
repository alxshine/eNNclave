cmake_minimum_required(VERSION 3.10)
project(eNNclave)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_library(backend_native SHARED
        backend/generated/native_forward.cpp
        inc/nn.h backend/native/src/output.cpp
        inc/IParameterLoader.h
        backend/native/src/nativeParameterLoader.cpp
        backend/native/inc/nativeParameterLoader.h
        core/nn.cpp)
target_include_directories(backend_native PUBLIC inc)
target_include_directories(backend_native PRIVATE backend/native/inc)
set_target_properties(backend_native PROPERTIES
        CXX_STANDARD 14)


add_library(frontend_python SHARED
        frontend/python/interoperability/frontend_python.c)
target_include_directories(frontend_python PRIVATE inc /usr/include/python3.8)
if (DEFINED ENV{SGX_SDK})
        target_compile_definitions(frontend_python PRIVATE SGX_SDK)
endif()
set_target_properties(frontend_python PROPERTIES
        PREFIX "")

if (DEFINED ENV{SGX_SDK})
        message(STATUS "Configuring SGX backend")

        set(SGX_LIB64 $ENV{SGX_SDK}/lib64)
        set(TSTDC "sgx_tstdc")
        set(TCXX "sgx_tcxx")
        set(TRTS "sgx_trts_sim")
        set(CRYPTO "sgx_tcrypto")
        set(SERVICE "sgx_tservice_sim")
        set(URTS "sgx_urts_sim")
        set(ENCLAVE_LINK_SCRIPT "${CMAKE_SOURCE_DIR}/backend/sgx/trusted/enclave.lds")
        set(SGX_EDL "${CMAKE_SOURCE_DIR}/backend/sgx/trusted/enclave.edl")
        set(ENCLAVE_C_FLAGS "-nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections")
        set(ENCLAVE_LINK_FLAGS "-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L${SGX_LIB64} -Wl,--whole-archive -l${TRTS} -Wl,--no-whole-archive -Wl,--start-group -l${TSTDC} -l${TCXX} -l${CRYPTO} -l${SERVICE} -Wl,--end-group -Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined -Wl,-pie,-eenclave_entry -Wl,--export-dynamic -Wl,--defsym,__ImageBase=0 -Wl,--gc-sections -Wl,--version-script=${ENCLAVE_LINK_SCRIPT}")
        
        find_library(TRTS_SIM sgx_trts_sim ${SGX_LIB64})
        find_library(STD_C sgx_tstdc ${SGX_LIB64})
        find_library(SGX_FS sgx_tprotected_fs ${SGX_LIB64})
        #TODO: exit with error if libraries not found
        # find_library(URTS_SIM sgx_urts_sim $ENV{SGX_SDK}/sdk_libs/)

        find_program(EDGER8R sgx_edger8r)
        add_custom_target(backend_sgx_generated COMMAND ${EDGER8R} --trusted-dir ${CMAKE_BINARY_DIR}/sgx/trusted/ --untrusted-dir ${CMAKE_BINARY_DIR}/sgx/untrusted/ --search-path ${CMAKE_SOURCE_DIR}/backend/sgx/trusted --search-path $ENV{SGX_SDK}/include ${CMAKE_SOURCE_DIR}/backend/sgx/trusted/enclave.edl
        BYPRODUCTS sgx/trusted/enclave_t.c;sgx/trusted/enclave_t.h;sgx/untrusted/enclave_u.c;sgx/untrusted/enclave_u.h)
        find_program(SIGN sgx_sign)
        # TODO: exit with error if programs not found
        
        # edger8r must be executed at configure time to generate files, and at compile time to update files
        # execute_process(COMMAND ${EDGER8R} --trusted-dir ${CMAKE_BINARY_DIR}/sgx/trusted/ --untrusted-dir ${CMAKE_BINARY_DIR}/sgx/untrusted/ 
        # --search-path ${CMAKE_SOURCE_DIR}/backend/sgx/trusted --search-path $ENV{SGX_SDK}/include 
        # ${CMAKE_SOURCE_DIR}/backend/sgx/trusted/enclave.edl)

        add_library(backend_sgx_enclave_t OBJECT
                ${CMAKE_BINARY_DIR}/sgx/trusted/enclave_t.c)
        target_include_directories(backend_sgx_enclave_t PRIVATE
        ${CMAKE_BINARY_DIR}/sgx/trusted
        $ENV{SGX_SDK}/include
        $ENV{SGX_SDK}/include/tlibc
        $ENV{SGX_SDK}/include/libcxx)
        set_target_properties(backend_sgx_enclave_t PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_STANDARD 99
        COMPILE_FLAGS ${ENCLAVE_C_FLAGS})

        add_library(backend_sgx_trusted MODULE
                backend/generated/sgx_forward.cpp
                backend/sgx/trusted/src/sgxParameterLoader.cpp
                backend/sgx/trusted/src/output.cpp
                backend/sgx/trusted/src/encryption.cpp
                $<TARGET_OBJECTS:backend_sgx_enclave_t>
                core/nn.cpp)
        add_dependencies(backend_sgx_trusted backend_sgx_generated)
        target_include_directories(backend_sgx_trusted PUBLIC
                inc)
        target_include_directories(backend_sgx_trusted PRIVATE 
                ${CMAKE_BINARY_DIR}/sgx/trusted/
                $ENV{SGX_SDK}/include
                $ENV{SGX_SDK}/include/tlibc
                $ENV{SGX_SDK}/include/libcxx
                backend/sgx/trusted/inc
                backend/sgx/inc)

        set_target_properties(backend_sgx_trusted PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                CXX_STANDARD 14
                COMPILE_FLAGS "${ENCLAVE_C_FLAGS} -nostdinc++"
                LINK_DEPENDS ${ENCLAVE_LINK_SCRIPT})
        target_link_libraries(backend_sgx_trusted
        -Wl,--no-undefined;-nostdlib;-nodefaultlibs;-nostartfiles;-L${SGX_LIB64};-Wl,--whole-archive;${TRTS_SIM};-Wl,--no-whole-archive;-Wl,--start-group;-l${TSTDC};-l${TCXX};-l${CRYPTO};-l${SERVICE};${SGX_FS};-Wl,--end-group;-Wl,-Bstatic;-Wl,-Bsymbolic;-Wl,--no-undefined;-Wl,-pie,-eenclave_entry;-Wl,--export-dynamic;-Wl,--defsym,__ImageBase=0;-Wl,--gc-sections;-Wl,--version-script=${ENCLAVE_LINK_SCRIPT})
        # TODO: replace libraries with find_library

        add_custom_target(backend_sgx_signed
        DEPENDS backend_sgx_trusted
        COMMAND ${SIGN} sign -key ${CMAKE_SOURCE_DIR}/backend/sgx/trusted/debug_signing_key.pem -enclave $<TARGET_FILE:backend_sgx_trusted> -out ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libbackend_sgx_trusted.signed.so -config ${CMAKE_SOURCE_DIR}/backend/generated/sgx_config.xml
        BYPRODUCTS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libbackend_sgx_trusted.signed.so)

        # add_executable(app
        # App/App.cpp
        # ${CMAKE_BINARY_DIR}/sgx/untrusted/Enclave_u.c)
        # target_include_directories(app PRIVATE
        # ${CMAKE_BINARY_DIR}/sgx/untrusted/
        # $ENV{SGX_SDK}/include
        # ${CMAKE_SOURCE_DIR}/Include
        # ${CMAKE_SOURCE_DIR}/App)
        # target_link_libraries(app backend_sgx_trusted ${URTS_SIM})
else()
        message(WARNING "SGX SDK not found, skipping")
endif ()

# Download and unpack googletest at configure time
configure_file(googleTest_CMakelists.txt googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(core_tests backend/generated/core_tests.cpp core/nn.cpp)
target_include_directories(core_tests PRIVATE inc)
target_link_libraries(core_tests gmock_main)
#add_test(NAME core_test COMMAND core_tests)
cmake_minimum_required(VERSION 3.10)
project(eNNclave)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

project(backend_native)
include_directories(inc backend/native/inc)
add_library(backend_native SHARED
        backend/generated/native_forward.cpp
        inc/nn.h backend/native/src/output.cpp
        inc/IParameterLoader.h
        backend/native/src/nativeParameterLoader.cpp
        backend/native/inc/nativeParameterLoader.h
        core/nn.cpp)
set_target_properties(backend_native PROPERTIES
        CXX_STANDARD 14)

project(frontend_python)
include_directories(
        /usr/include/python3.8
        inc
)

if (DEFINED $ENV{SGX_SDK})
        message(STATUS "Configuring SGX backend")
        add_compile_definitions(SGX_SDK)

        set(SGX_LIB64 $ENV{SGX_SDK}/lib64)
        set(TSTDC "sgx_tstdc")
        set(TRTS "sgx_trts_sim")
        set(CRYPTO "sgx_tcrypto")
        set(SERVICE "sgx_tservice_sim")
        set(URTS "sgx_urts_sim")
        set(ENCLAVE_LINK_SCRIPT "${CMAKE_SOURCE_DIR}/backend/sgx/trusted/enclave.lds")

        find_program(EDGER8R sgx_edger8r)
        execute_process(
        COMMAND ${EDGER8R} --trusted-dir sgx/trusted/ --untrusted-dir sgx/untrusted/ --search-path ${CMAKE_SOURCE_DIR}/backend/sgx/trusted --search-path $ENV{SGX_SDK}/include ${CMAKE_SOURCE_DIR}/backend/sgx/trusted/enclave.edl)

        add_library(backend_sgx_trusted SHARED
        ${CMAKE_BINARY_DIR}/sgx/trusted/Enclave_t.c
        Enclave/Enclave.cpp)
        target_include_directories(backend_sgx_trusted PRIVATE 
        ${CMAKE_BINARY_DIR}/sgx/trusted/
        $ENV{SGX_SDK}/include
        $ENV{SGX_SDK}/include/tlibc
        ${CMAKE_SOURCE_DIR}/Include)
        find_library(TRTS_SIM sgx_trts_sim $ENV{SGX_SDK}/lib64)
        find_library(STD_C sgx_tstdc $ENV{SGX_SDK}/lib64)
        # TODO: exit with error if not found
        message(STATUS "trts simulator library: ${TRTS_SIM}")
        target_link_libraries(backend_sgx_trusted ${TRTS_SIM} ${STD_C})
        # target_compile_options(backend_sgx_trusted PRIVATE -fPIC)
        set_target_properties(backend_sgx_trusted PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                LINK_DEPENDS ${ENCLAVE_LINK_SCRIPT}
                LINK_FLAGS "-Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L${SGX_LIB64} \
                -Wl,--whole-archive -l${TRTS} -Wl,--no-whole-archive \
                -Wl,--start-group -l${TSTDC} -l${CRYPTO} -l${SERVICE} -Wl,--end-group \
                -Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
                -Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
                -Wl,--defsym,__ImageBase=0 -Wl,--gc-sections   \
                -Wl,--version-script=${ENCLAVE_LINK_SCRIPT}")

        find_program(SIGN sgx_sign)
        add_custom_command(TARGET backend_sgx_trusted
        POST_BUILD
        COMMAND ${SIGN} ARGS sign 
        -key ${CMAKE_SOURCE_DIR}/Enclave/Enclave_private_test.pem 
        -enclave $<TARGET_FILE:backend_sgx_trusted> 
        -out $<TARGET_FILE_DIR:backend_sgx_trusted>/libbackend_sgx_trusted_signed.so # TODO: use generator for filename
        -config ${CMAKE_SOURCE_DIR}/Enclave/Enclave.config.xml)

        add_executable(app
        App/App.cpp
        ${CMAKE_BINARY_DIR}/sgx/untrusted/Enclave_u.c)
        target_include_directories(app PRIVATE
        ${CMAKE_BINARY_DIR}/sgx/untrusted/
        $ENV{SGX_SDK}/include
        ${CMAKE_SOURCE_DIR}/Include
        ${CMAKE_SOURCE_DIR}/App)
        find_library(URTS_SIM sgx_urts_sim $ENV{SGX_SDK}/sdk_libs/)
        # TODO: exit with error if not found
        message(STATUS "urts simulator library: ${URTS_SIM}")
        target_link_libraries(app backend_sgx_trusted ${URTS_SIM})    
endif ()

add_library(frontend_python SHARED
        frontend/python/interoperability/frontend_python.c)
set_target_properties(frontend_python PROPERTIES
        PREFIX "")

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
target_link_libraries(core_tests gmock_main)
#add_test(NAME core_test COMMAND core_tests)
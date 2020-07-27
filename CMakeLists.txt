cmake_minimum_required(VERSION 3.16)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

project(backend_native)
include_directories(inc)
add_library(backend_native SHARED
        backend/generated/native_forward.c
        backend/native/output.c
        backend/native/parameters.c
        core/nn.c)


project(frontend_python)
include_directories(
        /usr/include/python3.8
        inc
)

if (DEFINED $ENV{SGX_SDK})
    add_compile_definitions(SGX_SDK)
endif ()

add_library(frontend_python SHARED
        frontend/python/interoperability/frontend_python.c)
set_target_properties(frontend_python PROPERTIES
        PREFIX "")

if (DEFINED $ENV{SGX_SDK})
    add_dependencies(frontend_python backend_sgx)
    target_link_libraries(frontend_python backend_sgx)
endif ()
add_dependencies(frontend_python backend_native)
target_link_libraries(frontend_python backend_native)




project(core_test)
include_directories(inc backend/native/tests)
add_executable(core_test
        backend/native/tests/test_core.c
        backend/native/tests/assert.c
        backend/native/tests/util.c
        backend/native/tests/multiply.c
        backend/native/tests/add.c
        backend/native/tests/sep_conv1.c
        backend/native/tests/conv2.c
        backend/native/tests/depthwise_conv2.c
        backend/native/tests/relu.c
        backend/native/tests/global_average_pooling1.c
        backend/native/tests/global_average_pooling2.c
        backend/native/tests/max_pool1.c
        backend/native/tests/max_pool2.c
        backend/native/tests/zero_pad2.c backend/native/tests/tests.h)

add_dependencies(core_test backend_native)
target_link_libraries(core_test backend_native)
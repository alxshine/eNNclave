cmake_minimum_required(VERSION 3.16)
project(frontend_python)

set(CMAKE_CXX_STANDARD 14)

include_directories(
        /usr/include/python3.8
        inc
)

if (DEFINED $ENV{SGX_SDK})
    add_compile_definitions(SGX_SDK)
endif ()

add_library(frontend_python SHARED
        interop/pymatutilmodule.c)
set_target_properties(frontend_python PROPERTIES
        PREFIX "")

if (DEFINED $ENV{SGX_SDK})
    add_dependencies(frontend_python eNNclaveSgx)
endif ()


project(backend_native)
set(CMAKE_CXX_STANDARD 14)

include_directories(inc)
add_library(backend_native SHARED
        lib/native/forward.c
        lib/native/native_nn.c
        lib/native/output.c
        lib/native/parameters.c
        lib/common/matutil.c)

project(core_test)
set(CMAKE_CXX_STANDARD 14)

include_directories(inc)

add_executable(core_test
        lib/test_matutil.c
        lib/test/assert.c
        lib/test/util.c
        lib/test/multiply.c
        lib/test/add.c
        lib/test/sep_conv1.c
        lib/test/conv2.c
        lib/test/depthwise_conv2.c
        lib/test/relu.c
        lib/test/global_average_pooling1.c
        lib/test/global_average_pooling2.c
        lib/test/max_pool1.c
        lib/test/max_pool2.c
        lib/test/zero_pad2.c)

add_dependencies(core_test backend_native)
target_link_libraries(core_test backend_native)
cmake_minimum_required(VERSION 3.16)
project(eNNclavePythonModule)

set(CMAKE_CXX_STANDARD 14)

include_directories(
        /usr/include/python3.8
        inc
)

if (DEFINED $ENV{SGX_SDK})
    add_compile_definitions(SGX_SDK)
endif ()

add_library(eNNclavePythonModule SHARED
        interop/pymatutilmodule.c)

if (DEFINED $ENV{SGX_SDK})
    add_dependencies(eNNclavePythonModule eNNclaveSgx)
endif ()


project(eNNclaveNative)
set(CMAKE_CXX_STANDARD 14)

include_directories(inc)
add_library(eNNclaveNative SHARED
        lib/native/forward.c
        lib/native/native_nn.c
        lib/native/output.c
        lib/native/parameters.c
        lib/common/matutil.c)

project(eNNclaveLibraryTest)
set(CMAKE_CXX_STANDARD 14)

include_directories(inc)

add_executable(eNNclaveLibraryTest
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

add_dependencies(eNNclaveLibraryTest eNNclaveNative)
target_link_libraries(eNNclaveLibraryTest eNNclaveNative)
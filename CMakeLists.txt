cmake_minimum_required(VERSION 3.16)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

project(backend_native)
include_directories(inc backend/native/inc)
add_library(backend_native SHARED
        backend/generated/native_forward.cpp
        inc/nn.h backend/native/src/output.cpp
        inc/IParameterLoader.h
        backend/native/src/CxxParameterLoader.cpp
        backend/native/inc/CxxParameterLoader.h
        core/nn.cpp)
set_target_properties(backend_native PROPERTIES
        CXX_STANDARD 14)

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
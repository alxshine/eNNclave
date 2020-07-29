cmake_minimum_required(VERSION 3.16)
project(eNNclave)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

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
add_executable(core_tests backend/native/tests/core_tests.cpp core/nn.cpp)
target_link_libraries(core_tests gmock_main)
#add_test(NAME core_test COMMAND core_tests)
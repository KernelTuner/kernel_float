cmake_minimum_required(VERSION 3.22)

set (PROJECT_NAME kernel_float)
project(${PROJECT_NAME} CXX CUDA)

set(CMAKE_C_STANDARD 11)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE "${PROJECT_SOURCE_DIR}/include")

if (KERNEL_FLOAT_BUILD_TEST)
    add_subdirectory(tests)
endif()

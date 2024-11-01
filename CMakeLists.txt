cmake_minimum_required(VERSION 3.20)

set (PROJECT_NAME kernel_float)
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Validate and enable the appropriate language
if (NOT DEFINED KERNEL_FLOAT_LANGUAGE)
    set(KERNEL_FLOAT_LANGUAGE "CUDA")
endif()

if (KERNEL_FLOAT_LANGUAGE STREQUAL "CUDA")
    enable_language(CUDA)
    set(KERNEL_FLOAT_LANGUAGE_CUDA ON)
elseif (KERNEL_FLOAT_LANGUAGE STREQUAL "HIP")
    enable_language(HIP)
    set(KERNEL_FLOAT_LANGUAGE_HIP ON)
else()
    message(FATAL_ERROR "KERNEL_FLOAT_LANGUAGE must be either 'HIP' or 'CUDA'")
endif()

# Create an interface library for kernel_float
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE "${PROJECT_SOURCE_DIR}/include")

# Optionally build tests and examples if the corresponding flags are set
option(KERNEL_FLOAT_BUILD_TEST "Build kernel float tests" OFF)
option(KERNEL_FLOAT_BUILD_EXAMPLE "Build kernel float examples" OFF)

if (KERNEL_FLOAT_BUILD_TEST)
    add_subdirectory(tests)
endif()

if (KERNEL_FLOAT_BUILD_EXAMPLE)
    add_subdirectory(examples)
endif()

# Display configuration
message(STATUS "=== Kernel Float ===")
message(STATUS "Using GPU Language: ${KERNEL_FLOAT_LANGUAGE}")
message(STATUS "Building Tests: ${KERNEL_FLOAT_BUILD_TEST}")
message(STATUS "Building Examples: ${KERNEL_FLOAT_BUILD_EXAMPLE}")
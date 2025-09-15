cmake_minimum_required(VERSION 3.16)

project(tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

file(GLOB FILES *.cu)
add_executable(kernel_float_tests ${FILES})
target_link_libraries(kernel_float_tests PRIVATE kernel_float)

add_subdirectory(Catch2)
target_link_libraries(kernel_float_tests PRIVATE Catch2::Catch2WithMain)

if(${KERNEL_FLOAT_LANGUAGE_CUDA})
    find_package(CUDA REQUIRED)
    target_include_directories(kernel_float_tests PRIVATE ${CUDA_TOOLKIT_INCLUDE})

    target_compile_options(kernel_float_tests PRIVATE "-ftime-report -ftime-report-details")
    target_compile_options(kernel_float_tests PRIVATE "--extended-lambda")

    # Default to 80 if user hasn't specified
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80")
    endif()

    set_target_properties(kernel_float_tests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(${KERNEL_FLOAT_LANGUAGE_HIP})
    set_source_files_properties(${FILES} PROPERTIES LANGUAGE HIP)
endif()

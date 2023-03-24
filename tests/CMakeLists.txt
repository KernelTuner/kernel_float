cmake_minimum_required(VERSION 3.16)

project(tests)
set(CMAKE_CXX_STANDARD 17)

file(GLOB FILES *.cu)
add_executable(kernel_float_tests ${FILES})
target_link_libraries(kernel_float_tests PRIVATE kernel_float)
target_compile_options(kernel_float_tests PRIVATE "--extended-lambda")
set_target_properties(kernel_float_tests PROPERTIES CUDA_ARCHITECTURES "80")

target_compile_options(kernel_float_tests PRIVATE "-ftime-report -ftime-report-details")

add_subdirectory(Catch2)
target_link_libraries(kernel_float_tests PRIVATE Catch2::Catch2WithMain)

find_package(CUDA REQUIRED)
target_include_directories(kernel_float_tests PRIVATE ${CUDA_TOOLKIT_INCLUDE})

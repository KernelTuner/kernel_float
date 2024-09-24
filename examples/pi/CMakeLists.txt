cmake_minimum_required(VERSION 3.17)

set (PROJECT_NAME kernel_float_pi)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/main.cu")
target_link_libraries(${PROJECT_NAME} kernel_float)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80")

find_package(CUDA REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_TOOLKIT_INCLUDE})

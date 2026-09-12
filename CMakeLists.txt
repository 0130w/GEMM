cmake_minimum_required(VERSION 3.24)
project(shared_mem_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(gemm
  shared_mem_gemm.cu
  load_tile.cuh
  store_tile.cuh
)

target_include_directories(gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(gemm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

target_link_libraries(gemm PRIVATE CUDA::cudart)
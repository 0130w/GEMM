add_executable(naive_gemm_test naive_gemm_test.cu)
target_link_libraries(naive_gemm_test PRIVATE gemm_headers CUDA::cublas)

add_executable(shared_mem_gemm_test shared_mem_gemm_test.cu)
target_link_libraries(shared_mem_gemm_test PRIVATE gemm_headers CUDA::cublas)

add_test(NAME naive_gemm_test COMMAND naive_gemm_test)
add_test(NAME shared_mem_gemm_test COMMAND shared_mem_gemm_test)
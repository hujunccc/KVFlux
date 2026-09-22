add_executable(kvflux_tests block_manager_test.cpp)
target_link_libraries(kvflux_tests PRIVATE KVFlux::kvflux)
add_test(NAME kvflux_tests COMMAND kvflux_tests)

add_executable(kvflux_layout_tests kv_layout_test.cpp)
target_link_libraries(kvflux_layout_tests PRIVATE KVFlux::kvflux)
add_test(NAME kvflux_layout_tests COMMAND kvflux_layout_tests)

if(KVFLUX_ENABLE_CUDA)
  add_executable(kvflux_gpu_tests gpu_memory_pool_test.cpp)
  target_link_libraries(kvflux_gpu_tests PRIVATE KVFlux::kvflux)
  add_test(NAME kvflux_gpu_tests COMMAND kvflux_gpu_tests)
  set_tests_properties(kvflux_gpu_tests PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
endif()

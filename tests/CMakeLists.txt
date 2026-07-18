# tests/CMakeLists.txt — plain-C++ test executables registered with ctest.
#
# No external framework: each file is a standalone program that returns 0 on
# success and aborts (nonzero) on the first failed CHECK. Each links the shared
# egg_sta library so it drives the real staCpu / staGpu / generateLayeredDag.
function(egg_add_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE egg_sta)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

egg_add_test(test_hand_computed        test_hand_computed.cpp)
egg_add_test(test_generator_invariants test_generator_invariants.cpp)
egg_add_test(test_oracle_consistency   test_oracle_consistency.cpp)
egg_add_test(test_edge_cases           test_edge_cases.cpp)
egg_add_test(test_corner_replay        test_corner_replay.cpp)

# Real ISCAS-85 netlists: build the target and register ONE test that passes the
# bundled circuits dir as argv[1] (an absolute path, so it resolves from the build dir).
add_executable(test_bench_circuits test_bench_circuits.cpp)
target_link_libraries(test_bench_circuits PRIVATE egg_sta)
target_include_directories(test_bench_circuits PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_bench_circuits
         COMMAND test_bench_circuits ${CMAKE_SOURCE_DIR}/bench/circuits)

# End-to-end smoke test: the shipped binary runs a small sweep and exits 0.
# With no GPU it takes the honest CPU-fallback path (still exit 0); with a device
# it additionally runs the oracle check and returns nonzero on any mismatch.
add_test(NAME sta_binary_smoke COMMAND sta 8 64)

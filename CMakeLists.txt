# CMakeLists.txt — configure + build the `sta` binary and its test suite.
#
# This lives alongside the hand-written Makefile (which still works); CMake adds
# CUDA auto-detection and a ctest-registered test suite. Core STA is pure C++17,
# so a plain g++ box (GitHub-hosted runners, dev machines with no driver) builds
# and tests everything; a box with nvcc additionally compiles the GPU kernel.
cmake_minimum_required(VERSION 3.18)
project(eda_graph_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenMP powers the fair multi-threaded CPU baseline (staCpuParallel). Optional:
# without it the pragmas are ignored and the baseline runs correctly, serially.
find_package(OpenMP QUIET COMPONENTS CXX)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- CUDA detection ---------------------------------------------------------
# Detect a CUDA toolchain without hard-requiring one. CheckLanguage(CUDA) probes
# for nvcc (on PATH or via CUDACXX / CMAKE_CUDA_COMPILER). When present we enable
# the CUDA language and compile the real kernel (src/sta_gpu.cu); when absent we
# compile the CPU-only stub (src/sta_gpu_stub.cpp). Either way the public API is
# identical and honest: with no device at runtime staGpu sets ranGpu=false and
# returns the CPU reference — it never fabricates a GPU result.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
    endif()
    set(EGG_GPU_SOURCE src/sta_gpu.cu)
    set(EGG_HAVE_CUDA ON)
    message(STATUS "eda-graph-gpu: CUDA toolchain found (${CMAKE_CUDA_COMPILER})")
    message(STATUS "eda-graph-gpu: compiling GPU kernel src/sta_gpu.cu")
else()
    set(EGG_GPU_SOURCE src/sta_gpu_stub.cpp)
    set(EGG_HAVE_CUDA OFF)
    message(STATUS "eda-graph-gpu: no CUDA toolchain — CPU-only build")
    message(STATUS "eda-graph-gpu: compiling stub src/sta_gpu_stub.cpp")
endif()

# --- Core library -----------------------------------------------------------
# Bundle the CPU reference + generator + the chosen GPU source into one library
# that both the `sta` binary and the test executables link against, so the tests
# exercise the exact same staCpu / staGpu / generateLayeredDag / maxAbsDiff code.
add_library(egg_sta STATIC src/sta_cpu.cpp src/sta_cpu_mt.cpp src/circuit.cpp ${EGG_GPU_SOURCE})
target_include_directories(egg_sta PUBLIC include)
if(EGG_HAVE_CUDA)
    set_target_properties(egg_sta PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(egg_sta PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "eda-graph-gpu: OpenMP found — multi-core CPU baseline enabled")
else()
    message(STATUS "eda-graph-gpu: OpenMP not found — CPU baseline runs single-threaded")
endif()

# --- Binaries ---------------------------------------------------------------
add_executable(sta src/main.cpp)
target_link_libraries(sta PRIVATE egg_sta)

# Reads real ISCAS-85 circuits and profiles their topology vs the synthetic generator.
add_executable(profile src/profile_main.cpp)
target_link_libraries(profile PRIVATE egg_sta)

# --- Tests ------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# eda-graph-gpu — build the STA primitive.
#   make        # GPU build via nvcc (runs on GPU if present, else CPU fallback)
#   make cpu    # CPU-only build via g++ (no CUDA toolchain needed)
CUDA   ?= /usr/local/cuda
NVCC   ?= $(CUDA)/bin/nvcc
ARCH   ?= sm_90
CXX    ?= g++
CXXFLAGS := -O2 -std=c++17 -Iinclude -fopenmp

CORE := src/sta_cpu.cpp src/sta_cpu_mt.cpp src/circuit.cpp

all: sta profile
sta: src/main.cpp $(CORE) src/sta_gpu.cu
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -Iinclude -Xcompiler -fopenmp $^ -o $@

profile: src/profile_main.cpp $(CORE) src/sta_gpu.cu
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -Iinclude -Xcompiler -fopenmp $^ -o $@

cpu: src/main.cpp $(CORE) src/sta_gpu_stub.cpp
	$(CXX) $(CXXFLAGS) $^ -o sta

clean:
	rm -f sta profile *.o
.PHONY: all cpu clean

# CS5379 GPU Programming Lecture - Build System
# ==============================================
# Builds all versions of vector addition (v0-v6)
# Usage: make [target]
#   make all    - Build all versions
#   make clean  - Remove executables
#   make help   - Show this help

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11
CUDA_FLAGS = -O3 -std=c++11 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90

# Source files and executables
SOURCES_CPP = matrixMul_cpu.cpp
SOURCES_CU = matrixMul_gpu_v1.cu matrixMul_gpu_v2.cu matrixMul_gpu_v3.cu matrixMul_gpu_v4.cu
EXECUTABLES = matrixMul_gpu_v1.exe matrixMul_gpu_v2.exe matrixMul_gpu_v3.exe matrixMul_gpu_v4.exe

# Default target
all: $(EXECUTABLES)
	@echo "✅ All versions compiled successfully!"
	@echo "   ./matrixMul_cpu.exe  # CPU baseline"
	@echo "   ./matrixMul_gpu_v1.exe  # Single Thread"
	@echo "   ./matrixMul_gpu_v2.exe  # Multiple threads"
	@echo "   ./matrixMul_gpu_v3.exe  # Multiple blocks"
	@echo "   ./matrixMul_gpu_v4.exe  # Multiple blocks with optimization"

# CPU version (C++)
matrixMul_cpu.exe: matrixMul_cpu.cpp
	@echo "🖥️  Compiling CPU version (v0)..."
	$(CXX) $(CXXFLAGS) $< -o $@

# CUDA versions
matrixMul_gpu_v%.exe: matrixMul_gpu_v%.cu cuda_error_check.h
	@echo "🚀 Compiling CUDA version (v$*)..."
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# Individual targets for explicit building
app0: matrixMul_cpu.exe
app1: matrixMul_gpu_v1.exe
app2: matrixMul_gpu_v2.exe
app3: matrixMul_gpu_v3.exe
app4: matrixMul_gpu_v4.exe

# Cleanup
clean:
	@echo "🧹 Cleaning up executables..."
	rm -f $(EXECUTABLES)
	@echo "✅ Clean complete!"

# Help information
help:
	@echo "CS5379 GPU Programming Lecture - Build System"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all     - Build all versions (default)"
	@echo "  app0    - Build CPU version only"
	@echo "  app1-6  - Build specific CUDA version"
	@echo "  clean   - Remove all executables"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make           # Build everything"
	@echo "  make app6      # Build only v6 (explicit memory)"
	@echo "  make clean all # Clean and rebuild"

# Utility targets
check-cuda:
	@echo "🔍 Checking CUDA installation..."
	@which nvcc > /dev/null && echo "✅ NVCC found: $$(nvcc --version | grep release)" || echo "❌ NVCC not found"
	@which nvidia-smi > /dev/null && echo "✅ NVIDIA driver found" || echo "❌ NVIDIA driver not found"

info:
	@echo "📊 Build Configuration:"
	@echo "  C++ Compiler: $(CXX)"
	@echo "  CUDA Compiler: $(NVCC)"
	@echo "  C++ Flags: $(CXXFLAGS)"
	@echo "  CUDA Flags: $(CUDA_FLAGS)"
	@echo "  Executables: $(EXECUTABLES)"

# Phony targets
.PHONY: all clean help check-cuda info app0 app1 app2 app3 app4 app5 app6

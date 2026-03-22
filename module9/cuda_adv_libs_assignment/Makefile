# CUDA toolkit path
CUDA_PATH ?= /usr/local/cuda

# Compilers
NVCC := $(CUDA_PATH)/bin/nvcc
HOST_COMPILER ?= g++

# Detect GPU architecture
GPU_ARCH ?= sm_70

# Directories
SRC_DIR   := src
INC_DIR   := include
OBJ_DIR   := obj

# Output binary
TARGET := ablation_engine

# Source files
SOURCES := $(SRC_DIR)/main.cu         \
           $(SRC_DIR)/transformer.cu   \
           $(SRC_DIR)/ablation.cu      \
           $(SRC_DIR)/baseline.cu      \
           $(SRC_DIR)/batched_engine.cu \
           $(SRC_DIR)/benchmark.cu     \
           $(SRC_DIR)/validation.cu

# Object files
OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SOURCES))

# Compiler flags
NVCCFLAGS := -std=c++14 -O2 -arch=$(GPU_ARCH)
NVCCFLAGS += -I$(INC_DIR)
NVCCFLAGS += --expt-extended-lambda
NVCCFLAGS += -Xcompiler -Wall

# Linker flags (cuBLAS)
LDFLAGS := -lcublas

# Thrust is header-only (no extra linking)

.PHONY: all run benchmark validate baseline batched clean help

all: $(TARGET)
	@echo "Build complete: ./$(TARGET)"

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

run: $(TARGET)
	./$(TARGET) --all

benchmark: $(TARGET)
	./$(TARGET) --benchmark

validate: $(TARGET)
	./$(TARGET) --validate

baseline: $(TARGET)
	./$(TARGET) --baseline

batched: $(TARGET)
	./$(TARGET) --batched

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

help:
	@echo "Targets:"
	@echo "  all        Build (default)"
	@echo "  run        Build + run --all"
	@echo "  benchmark  Benchmark sweep"
	@echo "  validate   Correctness check"
	@echo "  baseline   Sequential baseline"
	@echo "  batched    Batched engine"
	@echo "  clean      Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_PATH  (default: /usr/local/cuda)"
	@echo "  GPU_ARCH   (default: sm_70)"

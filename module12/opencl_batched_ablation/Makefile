# Makefile — OpenCL Batched Causal Ablation Engine
#
# Supports: Linux (NVIDIA/AMD/Intel), macOS, WSL.
# Adjust OPENCL_INC and OPENCL_LIB for your system.

# ── Compiler and flags ───────────────────────────
CXX      := g++
CXXFLAGS := -std=c++11 -O2 -Wall -Wextra -Iinclude

# ── OpenCL paths (override on command line) ──────
# NVIDIA:  OPENCL_INC=/usr/local/cuda/include
#          OPENCL_LIB=/usr/local/cuda/lib64
# Intel:   OPENCL_INC=/opt/intel/opencl/include
# macOS:   uses -framework OpenCL (no paths needed)
OPENCL_INC ?= /usr/local/cuda/include
OPENCL_LIB ?= /usr/local/cuda/lib64

# ── Platform detection ───────────────────────────
UNAME := $(shell uname -s 2>/dev/null || echo Win)
ifeq ($(UNAME),Darwin)
    CL_FLAGS  :=
    CL_LDFLAGS := -framework OpenCL
else
    CL_FLAGS   := -I$(OPENCL_INC)
    CL_LDFLAGS := -L$(OPENCL_LIB) -lOpenCL
endif

# ── Source files ─────────────────────────────────
SRC_DIR := src
SRCS    := $(SRC_DIR)/main.cpp         \
           $(SRC_DIR)/cl_setup.cpp     \
           $(SRC_DIR)/device_info.cpp  \
           $(SRC_DIR)/ablation_engine.cpp

OBJS    := $(SRCS:.cpp=.o)
TARGET  := ablation_engine

# ── Default target ───────────────────────────────
.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(CL_FLAGS) -o $@ $^ \
		$(CL_LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CL_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

help:
	@echo "Targets:"
	@echo "  all     - Build $(TARGET)"
	@echo "  clean   - Remove build artifacts"
	@echo ""
	@echo "Override OpenCL paths:"
	@echo "  make OPENCL_INC=/path OPENCL_LIB=/path"

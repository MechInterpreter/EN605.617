NVCC    := nvcc
TARGET  := mem_all
SRC     := main.cu

CXXSTD   := -std=c++17
OPTFLAGS := -O3
ARCHFLAGS :=

NVCCFLAGS := $(OPTFLAGS) $(CXXSTD) $(ARCHFLAGS)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

rebuild: clean all

run:
	./$(TARGET)
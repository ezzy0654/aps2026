CXX = g++
NVCC = /usr/local/cuda/bin/nvcc

# APSS25-style separation of host/CUDA compiler flags and linker flags.
CFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native -fopenmp -pthread \
         -I/usr/local/cuda/include -Iinclude
CUDA_CFLAGS = -std=c++17 -O3 -arch=sm_86 \
              -Xcompiler=-Wall -Xcompiler=-Wextra \
              -Xcompiler=-march=native \
              -Xcompiler=-fopenmp -Xcompiler=-pthread \
              -I/usr/local/cuda/include -Iinclude
# nvcc is the linker driver here, so forward pthread to its host linker.
LDFLAGS = -Xcompiler=-pthread -L/usr/local/cuda/lib64
LDLIBS = -lstdc++ -lcudart -lm -lgomp

# Tensor-Core code is opt-in; the default build remains the FP32 path.
ifeq ($(USE_TC),1)
CUDA_CFLAGS += -DUSE_TC
endif

TARGET = main
CPP_OBJS = $(patsubst src/%.cpp,obj/%.o,$(wildcard src/*.cpp))
CU_OBJS = $(patsubst src/%.cu,obj/%.o,$(wildcard src/*.cu))
OBJECTS = $(CPP_OBJS) $(CU_OBJS)

all: $(TARGET)

$(TARGET): create_obj $(OBJECTS)
	$(NVCC) $(CUDA_CFLAGS) -o $@ $(OBJECTS) $(LDFLAGS) $(LDLIBS)

create_obj:
	mkdir -p obj

# main.cpp uses CUDA runtime APIs, so compile it through nvcc.
obj/main.o: src/main.cpp
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $<

obj/%.o: src/%.cpp
	$(CXX) $(CFLAGS) -c -o $@ $<

obj/%.o: src/%.cu
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $<

clean:
	rm -rf $(TARGET) $(OBJECTS)

.PHONY: all clean create_obj

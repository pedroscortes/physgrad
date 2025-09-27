NVCC = nvcc
CXX = g++
CUDA_PATH = /usr/local/cuda-12.1

NVCC_FLAGS = -std=c++17 --use_fast_math -lineinfo --extended-lambda -arch=sm_75
CXX_FLAGS = -std=c++17 -O3 -Wall
INCLUDES = -I$(CUDA_PATH)/include -I./src
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lm -lGL -lglfw

SOURCES_CUDA = src/simulation.cu
SOURCES_CPP = src/main.cpp
SOURCES_CONSOLE = src/console.cpp
OBJECTS = $(SOURCES_CUDA:.cu=.o) $(SOURCES_CPP:.cpp=.o)
OBJECTS_CONSOLE = $(SOURCES_CUDA:.cu=.o) $(SOURCES_CONSOLE:.cpp=.o)

all: physgrad

physgrad: $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

console: $(OBJECTS_CONSOLE)
	$(NVCC) $(NVCC_FLAGS) -o physgrad_console $^ -L$(CUDA_PATH)/lib64 -lcudart -lm

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

tests: console
	cd tests && $(MAKE)

clean:
	rm -f src/*.o physgrad physgrad_console

.PHONY: all console tests clean
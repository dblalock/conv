#!/usr/bin/make

# CC := clang
CXX := g++
NVCC := nvcc
CXXFLAGS := -O3 -std=c++14
NVCCFLAGS := -O3 -std=c++14 -Icuda-api-wrappers/src
# INCLUDE_FLAGS := -Icuda-api-wrappers/src

# LDFLAGS := -lgtest -lgtest_main -lpthread -L/usr/lib
LDFLAGS := -lgtest -lpthread -L/usr/lib

GTEST_DIR := /usr

all: vec_add.out naive_conv.out
	@echo making all...

# naive_conv.out: naive_conv.o
# 	@echo ------------------------ Compiling $@ ...
# 	$(CXX) $(LDFLAGS) $< -o $@

# naive_conv.out: naive_conv.cpp
# 	@echo ------------------------ Compiling $@ ...
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -c -o $@

%.out: %.cu
	@echo ------------------------ Compiling $@ ...
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cu.o: %.cu
	@echo ------------------------ Compiling $@ ...
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

%.o: %.cpp
	@echo ------------------------ Compiling $@ ...
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.out: %.cpp
	@echo ------------------------ Linking $@ ...
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -c -o $@

# %.out

.PHONY: clean
clean:
	rm -rf *.out */*.out *.o */*.o *.so */*.so

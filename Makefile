#!/usr/bin/make

# CC := clang
CXX := g++
NVCC := nvcc
# CXXFLAGS := -std=c++14 -DEIGEN_MPL2_ONLY
CXXFLAGS := -Ofast -march=native -std=c++14 -DEIGEN_MPL2_ONLY
NVCCFLAGS := -O3 -std=c++14 -Icuda-api-wrappers/src
# INCLUDE_FLAGS := -Icuda-api-wrappers/src

# LDFLAGS := -lgtest -lgtest_main -lpthread -L/usr/lib
LDFLAGS := -lgtest -lbenchmark -lpthread -L/usr/lib

TEST_FILES := tests/tests_main.o tests/test_naive_conv.o
BENCHMARK_FILES := bench/benchmarks_main.o bench/benchmark_dummy.o

TESTS_BINARY := bin/tests.out
BENCHMARKS_BINARY := bin/bench.out

# TESTS_MAIN_OBJ = test/tests_main.o

all: tests benchmarks vec_add
	@echo making all...

tests/test_naive_conv.o: tests/test_naive_conv.cpp src/naive_conv.hpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

# naive_conv.out: naive_conv.o
# 	@echo ------------------------ Compiling $@ ...
# 	$(CXX) $(CXXFLAGS) naive_conv.o $(LDFLAGS) -o naive_conv.out

# naive_conv.o: naive_conv.cpp
# 	@echo ------------------------ Compiling $@ ...
# 	$(CXX) $(CXXFLAGS) $< -c -o $@

# tests: $(TEST_FILES) test_main.o
# .PHONY: tests

tests: $(TEST_FILES)
	$(CXX) $(CXXFLAGS) $(TEST_FILES) $(LDFLAGS) -o $(TESTS_BINARY)
# test_main.out: test_main.o
# test_main.o: $(TEST_FILES)

benchmarks: $(BENCHMARK_FILES)
	$(CXX) $(CXXFLAGS) $(BENCHMARK_FILES) $(LDFLAGS) -o $(BENCHMARKS_BINARY)

.PHONY: vec_add.out
vec_add: vec_add.out
	$(shell mv vec_add.out bin/)


%.out: %.cu
	@echo ------------------------ Compiling $@ ...
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.o: %.cu
	@echo ------------------------ Compiling $@ ...
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

# .cpp.o:
%.o: %.cpp
	@echo ------------------------ Compiling $@ ...
	$(CXX) $(CXXFLAGS) $< -c -o $@

# %.out: %.cpp
# 	@echo ------------------------ Compiling + Linking $@ ...
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $<
# 	# $(CXX) $(CXXFLAGS) $(LDFLAGS) $< -c -o $@

%.out: %.o
	@echo ------------------------ Compiling + Linking $@ ...
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

# %.out

.PHONY: clean
clean:
	rm -rf *.out */*.out *.o */*.o *.so */*.so

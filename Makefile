#!/usr/bin/make

# CC := clang
CXX := g++
NVCC := nvcc
CXXFLAGS := -O3 -std=c++14
NVCCFLAGS := -O3 -std=c++14 -Icuda-api-wrappers/src
# INCLUDE_FLAGS := -Icuda-api-wrappers/src

# LDFLAGS := -lgtest -lgtest_main -lpthread -L/usr/lib
LDFLAGS := -lgtest -lbenchmark -lpthread -L/usr/lib

TEST_FILES := test_naive_conv.o
BENCHMARK_FILES := benchmark_dummy.o

TESTS_BINARY := tests.out
BENCHMARKS_BINARY := bench.out

all: tests benchmarks vec_add.out
	@echo making all...

# naive_conv.out: naive_conv.o
# 	@echo ------------------------ Compiling $@ ...
# 	$(CXX) $(CXXFLAGS) naive_conv.o $(LDFLAGS) -o naive_conv.out

# naive_conv.o: naive_conv.cpp
# 	@echo ------------------------ Compiling $@ ...
# 	$(CXX) $(CXXFLAGS) $< -c -o $@

# tests: $(TEST_FILES) test_main.o
# .PHONY: tests

tests: tests_main.o $(TEST_FILES)
	$(CXX) $(CXXFLAGS) tests_main.o $(TEST_FILES) $(LDFLAGS) -o $(TESTS_BINARY)
# test_main.out: test_main.o
# test_main.o: $(TEST_FILES)

benchmarks: benchmarks_main.o $(BENCHMARK_FILES)
	$(CXX) $(CXXFLAGS) benchmarks_main.o $(BENCHMARK_FILES) $(LDFLAGS) -o $(BENCHMARKS_BINARY)


%.out: %.cu
	@echo ------------------------ Compiling $@ ...
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.o: %.cu
	@echo ------------------------ Compiling $@ ...
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

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

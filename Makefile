#!/usr/bin/make

# CC := clang
CXX := g++
NVCC := nvcc
# CXXFLAGS := -std=c++14 -DEIGEN_MPL2_ONLY
# CXXFLAGS := -Ofast -march=native -std=c++14 -DEIGEN_MPL2_ONLY -fno-rtti
CXXFLAGS := -Ofast -march=native -std=c++14 -DEIGEN_MPL2_ONLY
NVCCFLAGS := -O3 -std=c++14 -Icuda-api-wrappers/src
# INCLUDE_FLAGS := -Icuda-api-wrappers/src

TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null)
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'  2>/dev/null)

LDFLAGS := -lgtest -lbenchmark -lpthread -L/usr/lib

TEST_FILES := tests/tests_main.o tests/test_direct_conv.o tests/test_catconv.o
TEST_FILES += tests/test_ksparse_interop.o tests/test_ksparse_conv.o
BENCHMARK_FILES := bench/benchmarks_main.o bench/benchmark_dummy.o

TESTS_BINARY := bin/tests.out
BENCHMARKS_BINARY := bin/bench.out

OPS_LIBS := lib/example_ops.so lib/catconv_ops.so
# OPS_LIBS := lib/example_ops.so lib/catconv_ops.so lib/pq_ops.so

all: tests benchmarks vec_add ops
	@echo making all...

# ================================================================ tests

tests/test_direct_conv.o: tests/test_direct_conv.cpp src/direct_conv.hpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

tests/test_catconv.o: tests/test_catconv.cpp src/catconv.hpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

tests/test_ksparse_interop.o.o: tests/test_ksparse_interop.cpp \
	src/ksparse_interop.hpp src/ksparse_interop_grad.hpp src/ksparse_util.hpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

tests/test_ksparse_interop.o.o: tests/test_ksparse_conv.cpp \
	src/ksparse_conv.hpp src/ksparse_conv_grad.hpp src/ksparse_util.hpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

tests: $(TEST_FILES)
	$(CXX) $(CXXFLAGS) $(TEST_FILES) $(LDFLAGS) -o $(TESTS_BINARY)
# test_main.out: test_main.o
# test_main.o: $(TEST_FILES)

# ================================================================ benchmarks

benchmarks: $(BENCHMARK_FILES)
	$(CXX) $(CXXFLAGS) $(BENCHMARK_FILES) $(LDFLAGS) -o $(BENCHMARKS_BINARY)

# ================================================================ ops

OP_BUILD_COMMAND = $(CXX) $(CXXFLAGS) -fPIC -shared ${TF_CFLAGS} ${TF_LFLAGS}

EXAMPLE_OPS_DEPS = src/example_ops.cpp
lib/example_ops.so: $(EXAMPLE_OPS_DEPS)
	$(OP_BUILD_COMMAND) $^ -o $@

CATCONV_DEPS := src/catconv_ops.cpp
lib/catconv_ops.so: $(CATCONV_DEPS) src/catconv.hpp
	$(OP_BUILD_COMMAND) $(CATCONV_DEPS) -o $@

# PQ_OPS_DEPS := src/pq_ops.cpp
# lib/pq_ops.so: $(PQ_OPS_DEPS)
# 	$(OP_BUILD_COMMAND) $(PQ_OPS_DEPS) -o $@


.PHONY: ops
ops: $(OPS_LIBS)
	@echo ------------------------ Making ops ...
	@echo

# ================================================================ cuda debug

.PHONY: vec_add.out
vec_add: vec_add.out
	$(shell mv vec_add.out bin/)

# ================================================================ misc

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

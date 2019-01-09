#!/usr/bin/make

NVCC := nvcc
INCLUDE_FLAGS := -Icuda-api-wrappers/src
NVCCFLAGS := -O3 -std=c++14 $(INCLUDE_FLAGS)

all: vec_add.out
	@echo making all...

%.out: %.cu
	@echo ------------------------ Compiling $@ ...
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cu.o: %.cu
	@echo ------------------------ Compiling $@ ...
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: clean
clean:
	rm -rf *.out */*.out *.o */*.o *.so */*.so

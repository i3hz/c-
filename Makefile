.PHONY: main

GCC = g++
NVCC = nvcc

main:
	@$(GCC) -o c++/main c++/main.cpp
	@./c++/main
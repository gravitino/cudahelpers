all: example

example: example.cu cuda_helpers.cuh
	nvcc -arch=sm_35 --expt-extended-lambda example.cu -o example -O3

clean:
	rm -rf example

Project Introduction:
			A two wheeled robot is to navigate in an environment filled with obstacles. The robot has to utilize the observations from noisy measurements recorded by sensors and localise itself in the environment. Particle filters are utilized for the same and this project is aimed to parallelise the computations involved in the same to reduce computation time.
This program is written in C++ and program parallelisation has been implemented by utilising functionalities of parallel processing libraries. This was implemented on two different libraries – CUDA; OPENMP .

Running the program:
Install CUDA/OPENMP
CUDA libraries used:   curand.h; curand_kernel.h
CUDA version used : 10.0
Download the programs in the folder - particlefilter_submission
To compile:
Serial code      : g++ -std=c++11 generator.cpp
OpenMp code:  g++ -std=c++11-fopenmpgenerator_omp.cpp
CUDA code      : nvcc-std=c++11 gen_cuda.cu
Output:
./a.out




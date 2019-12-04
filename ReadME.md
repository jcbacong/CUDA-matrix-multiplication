# Matrix multiplciation in CUDA C

The ```main.cu``` file performs simple matrix multiplication of two square matrices A and B using two cuda kernels, namely:

1. ```matmul_rec_glob``` for naive way of multiplying matrices using global memory

2. ```matmul_rec_shared``` for multiplying matrices using shared memory

In the experiments, the matrix dimension and tile width for the shared memory were varied; thus, the different ```*.exe``` programs inside the repository. The ```shared.cu``` file was an attempt to programatically adjust the tile width of the shared memory, but this was not possible as of this writing. A technical report is also included in the repo for further performance analysis. 

Should you have any comments on the report or code, please do message me on LinkedIn or email me at ```bacong.junelle@gmail.com```.
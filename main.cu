#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 64

void randomArray(int *A, int n, int k){
	srand((unsigned) time(NULL));
	for(int i=0; i< n*k; ++i)
		A[i] = ((int)rand()% 10) + 1;
	}

void printResults(int *h_matA, int *h_matB, int *h_matC, int n, int k, int m){
	printf("Matrix A:\n");
	for(int i=0; i< (n*k); i++){
		// int id = i + floor(i / (int)SQTILE_WIDTH )* (int)SQTILE_WIDTH;
		printf("%d	", h_matA[i]);
		if( (i+1) % k  == 0 ){
			printf("\n");
		}

	}
	printf("Matrix B:\n");
	for(int i=0; i< (k * m); i++){
		// int id = i + floor(i / (int)SQTILE_WIDTH )* (int)SQTILE_WIDTH;
		printf("%d	", h_matB[i]);
		if( (i+1) % m  == 0 ){
			printf("\n");
		}
	}

	printf("Matrix C:\n");
	for(int i=0; i< (n * m); i++){
		// int id = i + floor(i / (int)SQTILE_WIDTH )* (int)SQTILE_WIDTH;
		printf("%d	", h_matC[i]);
		if( (i+1) % m  == 0 ){
			printf("\n");
		}
	}
}

__global__ void matmul_rec_glob(int *a, int *b, int *c, int n, int k, int m) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if( col < m && row < n) {
    	int sum = 0;
        for(int i = 0; i < k; i++) {
            sum += a[row*k + i] * b[i*m + col];
        }
        c[row * m + col] = sum;
    }
}


__global__ void matmul_rec_shared(int *a, int *b, int *c, int n, int k, int m) {

	__shared__ int sA[TILE_WIDTH][TILE_WIDTH];
	__shared__ int sB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    int Pvalue = 0;

    for (int i = 0; i < (m-1)/TILE_WIDTH+1; ++i) {
       if (Row < n && i*TILE_WIDTH+tx < k)
          sA[ty][tx] = a[Row*k + i*TILE_WIDTH+tx];
       else
          sA[ty][tx] = 0;
        
       if (Col < m && i*TILE_WIDTH+ty < k)
          sB[ty][tx] = b[(i*TILE_WIDTH+ty)*m + Col];
       else
          sB[ty][tx] = 0;

       __syncthreads();
       for (int j = 0; j < TILE_WIDTH; ++j)
          Pvalue += sA[ty][j] * sB[j][tx];
       __syncthreads();
    }
    if (Row < n && Col < m)
       c[Row*m+Col] = Pvalue;
}


int main() {
	// Create matrices
	// A = nxk, B = kxm, C = nxm
	// printf("Enter valid dimension of matrices (A = nxk, B = kxm): \n");
	int dev;
	printf("[**] Input CUDA device no: ");
   	scanf("%d", &dev);
	dim dim0 = dim(10000,9000,10000),
		dim1 = dim(10000,10000,20000),
		dim2 = dim(10000,20000,30000),
		dim4 = dim(10000,20000,40000),
		dim5 = dim(20000,20000,50000),
		dim6 = dim(20000,30000,40000);
	
	dim DIM[6] = {dim0, dim1, dim2, dim4, dim5, dim6};
	// Multiprocessing constants
	
	// scanf("%d %d %d", &n, &k, &m);
	for(int j = 0; j<6; j++) {
		int n = DIM[j].n, k = DIM[j].k, m = DIM[j].m;
		
		unsigned int grid_rows = ceil(n / TILE_WIDTH) < 1 ? 1 : ceil(n/TILE_WIDTH);
		unsigned int grid_cols = ceil(m / TILE_WIDTH) < 1 ? 1 : ceil(m/TILE_WIDTH);
		const dim3 threadsPerBlock(TILE_WIDTH,TILE_WIDTH); 	// Must not exceed 1024 (max thread per block)
		const dim3 blocksPerGrid( grid_cols, grid_rows);

		// Initialize host matrices
		int *h_A, *h_B, *h_C;
		clock_t h_alloctime = clock();
	    h_A = (int*) malloc(sizeof(int)*k*n);
	    h_B = (int*) malloc(sizeof(int)*m*k);
	  	h_C = (int*) malloc(sizeof(int)*n*m);

	    printf("[**] Using tile width = %d...\n", TILE_WIDTH);
	    printf("[**] Creating matrix A with dimension %d x %d...\n", n,k);
			randomArray(h_A, n, k);
			printf("[**] Creating matrix B with dimension %d x %d...\n", k,m);
			randomArray(h_B, k, m);
			printf("[**] CPU Allocation time for the matrices: %.6f sec \n",(double)(clock()-h_alloctime)/CLOCKS_PER_SEC );

		// Allocate memory space on the device 
	    int *d_A, *d_B, *d_C;
	    cudaSetDevice(dev);
	    cudaError_t err;
	    clock_t d_alloctime = clock();

	    err = cudaMalloc((void **) &d_A, sizeof(int)*k*n);
	    if (err != cudaSuccess) 
			printf("Error: %s\n", cudaGetErrorString(err));

	    err = cudaMalloc((void **) &d_B, sizeof(int)*m*k);
	    if (err != cudaSuccess) 
			printf("Error: %s\n", cudaGetErrorString(err));

	    err = cudaMalloc((void **) &d_C, sizeof(int)*n*m);
	    if (err != cudaSuccess) 
			printf("Error: %s\n", cudaGetErrorString(err));

		err = cudaMemcpy(d_A, h_A, sizeof(int)*k*n, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) 
			printf("Error: %s\n", cudaGetErrorString(err));

		err = cudaMemcpy(d_B, h_B, sizeof(int)*k*m, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) 
			printf("Error: %s\n", cudaGetErrorString(err));

		printf("[**] GPU Allocation time for the matrices: %.6fsec \n",(double)(clock()-d_alloctime)/CLOCKS_PER_SEC );

		cudaEvent_t start,end;
		float ms, avems = 0.0;

		printf("     Starting kernel program 'matmul_rec_glob' execution\n");
		for(int i = 0; i<10; i++) {
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);

			matmul_rec_glob<<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, n, k, m);
			err = cudaGetLastError();
			if (err != cudaSuccess) 
			    printf("Error: %s\n", cudaGetErrorString(err));

			cudaEventRecord(end, 0);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&ms, start, end);

			// printf("\tIteration no. %d: %.6fsecs\n", i, ms);
			avems+=ms;
			cudaMemcpy(h_C, d_C, sizeof(int)*m*n, cudaMemcpyDeviceToHost); 
			cudaEventDestroy(start);
			cudaEventDestroy(end);
		}
		printf("\t   >>> Average kernel execution time: %.6fsec.\n", avems/10.0);
		// printResults(h_A, h_B, h_C, n, k, m);

		printf("     Starting kernel program 'matmul_rec_shared' execution\n");
		avems = 0.0;

		
		for(int i = 0; i<10; i++) {
			cudaEventCreate(&start);
			cudaEventCreate(&end);
			cudaEventRecord(start, 0);

			matmul_rec_shared<<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, n, k, m);
			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) 
			    printf("Error: %s\n", cudaGetErrorString(err));

			cudaEventRecord(end, 0);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&ms, start, end);

			// printf("\tIteration no. %d: %.6f sec\n", i, ms);
			avems+=ms;
			cudaMemcpy(h_C, d_C, sizeof(int)*m*n, cudaMemcpyDeviceToHost); 
			cudaEventDestroy(start);
			cudaEventDestroy(end);
		}
		printf("\t  >>> Average kernel execution time: %.6f sec.\n", avems/10.0);
		printf("[**] Freed memory. Done.\n\n");

		// printResults(h_A, h_B, h_C, n, k, m);

		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		free(h_A);
		free(h_B);
		free(h_C);
		// printf("[**] Freed memory. Done.\n");
	}

	return 0;
	
}
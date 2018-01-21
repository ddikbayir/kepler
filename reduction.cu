
#include <time.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>

void run_test();
void printArr(int *arr);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float blockReduceSum(float val) {

  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void reduction(float *in, float *out, int N, int s1, int s2, int splane, int dim1, int dim2, int planeCount)
{
	float sum =0;
	int cur_plane;

	int start = blockIdx.x * 8 * blockDim.x + threadIdx.x;
	int gridStride = blockDim.x * 8 * gridDim.x;

	//relative index and coordinates calculation
	int area = dim1 * dim2;
	
	int target = (start%dim1) * s1;
 	target += ((start / dim1) % dim2) * s2;
 	target += ((start/area)%planeCount) * splane;
	/*int tempDiv = area;
	/*
	for(int dimIter=0; dimIter<noDims; dimIter++)
	{
		if(dimIter != noDims-1)
		{
			dCoord = start / tempDiv % dimSizes[dimIter];
		}
	}
	*/
	int counter = 0;
	int quarter = dim2 / 8;
	quarter = quarter *s2;
	for(int i = start;
		i < N;
		i += gridStride)
	{
		sum = 0;
		
		//float sum =0;
		//calculate the first target index
 		

 		//determine which plane the thread is reducing
 		//cur_plane = (int)(i/area);

 		//printf("Test: tid= %d  target= %d target2= %d \n\n", i, target, target + (dim2/2 * s2));
 		
 		for(int iter=0; iter < 8; iter++)
 		{
 			sum += in[gridStride*counter + target + iter*quarter];

 		}
 		//__syncthreads();
 		/*
 		sum = in[gridStride*counter + target] + in[gridStride*counter + target + quarter] +
 				in[gridStride*counter + target + 2*quarter] + in[gridStride*counter+target+ 3*quarter];
 		*/		
 		//sum += in[i] + in[i+blockDim.x];
 		sum = blockReduceSum(sum);
		if(threadIdx.x == 0)
			out[counter*gridDim.x + blockIdx.x] = sum;
 		
 		counter++;
 		//sum += in[i] + in[i + blockDim.x];
	}
	
}



int main(void)
{
	{
		run_test();
	}
	return 0;
}

void printArr(int *arr)
{
	int i;
	printf("Stride values in order: ");
	for(i=0;i<=sizeof(arr)/sizeof(int);i++)
	{

		printf("%d ", arr[i]);
	}
	printf("\n\n");
}
void run_test()
{

	printf("%.3f\n", float(20/1000) );
	const int dim_len = 4;

	//dimension sizes
	int dims[dim_len] = {32,32,1024,256};
	//dimensions to reduce
	int rdims[2] = {0,1}; //x and y


	int strides[dim_len];

	strides[0] = 1;

	//total number of elements
	int N = dims[0];




	for(int i=1; i<dim_len; i++){
		strides[i] = dims[i-1] * strides[i-1]; 
		//update N
		N *= dims[i];
	}
	printf("Number of elements: %d\n\n", N);
	printArr(strides);

	//Allocate memory for in and out in host and fill in
	float *in, *out, *d_in, *d_out;
	//int *d_strides, *d_rdims, *d_dims;

	in = (float*)malloc(N*sizeof(float));

	
	int planeCount = 1024*256;//8192;//131072;

	out = (float*)malloc(planeCount*sizeof(float)); 
	srand(time(NULL));
	for(int i=0; i<N;i++)
	{
		in[i] = (float)rand() / (float)RAND_MAX; //float(i)/float(1000);//
	}
	/*
	for(int i=0;i<N;i++)
	{
		printf("%.1f ", in[i]);
	}
	*/
	printf("\n\n");
	//Allocate memory for in and out on device
	cudaMalloc(&d_in, N*sizeof(float));
	cudaMalloc(&d_out, planeCount*sizeof(float));
	

	//Event variables
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
	//Transfer host data to device

	gpuErrchk(cudaMemcpy(d_in, in, N*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_out, out, planeCount*sizeof(float), cudaMemcpyHostToDevice));


	int s1 = strides[rdims[0]];
	int s2 = strides[rdims[1]];
	int splane = strides[2];
	int dim1 = dims[rdims[0]];
	int dim2 = dims[rdims[1]];
	int noEls = 8;
	//Record kernel
	int noMeasures = 10; //number of measurements to take
	cudaEventRecord(start);
	for(int mesIter=0; mesIter<noMeasures;mesIter++)
	{
		reduction<<<4096,128>>>(d_in,d_out, N, s1, s2, splane, dim1, dim2, planeCount);
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms,start,stop);

	gpuErrchk(cudaMemcpy(out, d_out, planeCount*sizeof(float), cudaMemcpyDeviceToHost));
	ms = ms/noMeasures;
	double total = (N+planeCount)*4;

	double ebw = total/(ms*1e6);

	printf("EBW: %f\n", ebw);
	//Check errors
	printf("Strides: %d %d ", s1,s2);
	printf("Plane Stride: %d\n ", splane);
	printf("Plane Count: %d\n", planeCount);

	for(int i=0;i<4;i++)
	{
		
		printf("%.3f %d ", out[i], i);
	}
	printf("%.3f", out[63/*131071*/]);
	printf("\n");
	cudaFree(d_in);
	cudaFree(d_out);
}
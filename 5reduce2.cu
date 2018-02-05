
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
float warpReduceSum(double val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
float blockReduceSum(double val) {

  static __shared__ double shared[32]; // Shared mem for 32 partial sums
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

__global__ void reduction(double *in, double *out, int N, int s1, int s2, int s3, int splane, int dim1, int dim2, int dim3, int planeCount, int noEls)
{
	double sum =0;
	int cur_plane;
	
	int area = dim1*dim2*dim3;
	int start = blockIdx.x * noEls * blockDim.x + threadIdx.x;
	//int gridStride = splane * gridDim.x;
	int gridStride = splane * gridDim.x;
	
	//relative index and coordinates calculation
	int target = 0;
	
	target = (start%dim1) * s1;
 	target += ((start/ dim1) % dim2) * s2;
 	target += ((start/dim1*dim2) % dim3) * s3;
 	target += ((start/area)) * splane;
	
	
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
	int quarter = (blockDim.x%dim1) * s1;
	quarter += ((blockDim.x/dim1) %dim2) * s2;
	quarter = (((blockDim.x/(dim1*dim2))) %dim3) * s3;
	//quarter = quarter * s3;
	//int quarter = blockDim.x * s1;
	//quarter = quarter *s2;
	//int step = s1; //step of the first reduction dimension
	for(int i = target;
		counter < planeCount/gridDim.x;
		i += gridStride)
	{
		sum = 0;
		
		//float sum =0;
		//calculate the first target index
 		

 		/*
		target = (i%dim1) * s1;
 		target += ((i/ dim1) % dim2) * s2;
 		target += ((i/area)) * splane;
 		*/
 		//printf("Test: tid= %d  target= %d target2= %d \n\n", i, target, target + (dim2/2 * s2));
 		
 		for(int iter=0; iter < noEls; iter++)
 		{
 			sum += in[i + iter*quarter];

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

void run_test(int noEls, int noOfBlocks, int r1, int r2, int r3, int rplane, int dimen1, int dimen2, int dimen3, int dimen4, int dimen5)
{

	
	const int dim_len = 5;

	//dimension sizes
	int dims[dim_len] = {dimen1,dimen2,dimen3,dimen4,dimen5};
	//dimensions to reduce
	//int rdims[2] = {r1,r2}; //x and y
	int rdims[3] = {r1,r2,r3};

	int strides[dim_len];

	strides[0] = 1;

	//total number of elements
	int N = dims[0];




	for(int i=1; i<dim_len; i++){
		strides[i] = dims[i-1] * strides[i-1]; 
		//update N
		N *= dims[i];
	}
	//N = dimen1 * dimen2 * dimen3;
	printf("Number of elements: %d\n\n", N);
	printArr(strides);

	//Allocate memory for in and out in host and fill in
	double *in, *out, *d_in, *d_out;
	//int *d_strides, *d_rdims, *d_dims;

	in = (double*)malloc(N*sizeof(double));

	
	int planeCount = int(N)/(dims[rdims[0]]*dims[rdims[1]]*dims[rdims[2]]);//dims[rplane]*dims[4];//8192;//131072;

	out = (double*)malloc(planeCount*sizeof(double)); 
	srand(time(NULL));
	for(int i=0; i<N;i++)
	{
		if((i/128) %16384 == 16300)
		{
			in[i] =  double(i)/1000;//(float)rand() / (float)RAND_MAX;////
		}
		else
		{
			in[i] = double(i)/1000;//(float)rand() / (float)RAND_MAX;
		}
	}
	/*
	for(int i=0;i<N;i++)
	{
		printf("%.1f ", in[i]);
	}
	*/
	printf("\n\n");
	//Allocate memory for in and out on device
	cudaMalloc(&d_in, N*sizeof(double));
	cudaMalloc(&d_out, planeCount*sizeof(double));
	

	//Event variables
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
	//Transfer host data to device

	gpuErrchk(cudaMemcpy(d_in, in, N*sizeof(double), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_out, out, planeCount*sizeof(double), cudaMemcpyHostToDevice));


	int s1 = strides[rdims[0]];
	int s2 = strides[rdims[1]];
	int s3 = strides[rdims[2]];
	
	int splane = strides[rplane];
	int dim1 = dims[rdims[0]];
	int dim2 = dims[rdims[1]];
	int dim3 = dims[rdims[2]];
	
	//int noElems = noEls;
	//Record kernel
	int noMeasures = 100; //number of measurements to take
	cudaEventRecord(start);
	for(int mesIter=0; mesIter<noMeasures;mesIter++)
	{
		reduction<<<noOfBlocks,((dim1*dim2*dim3)/noEls)>>>(d_in,d_out, N, s1,s2,s3, splane, dim1,dim2,dim3, planeCount, noEls);
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms,start,stop);

	gpuErrchk(cudaMemcpy(out, d_out, planeCount*sizeof(double), cudaMemcpyDeviceToHost));
	ms = ms/noMeasures;
	double total = (N)*8;

	double ebw = total/(ms*1e6);

	printf("EBW: %f\n", ebw);
	//Check errors
	printf("Strides: %d  ", s1);
	printf("Plane Stride: %d\n ", splane);
	printf("Plane Count: %d\n", planeCount);

	for(int i=0;i<4;i++)
	{
		
		printf("%d : %.3f  ", i, out[i]);
	}
	double sizeOut = sizeof(out);
	double sizeD = sizeof(double);
	int lengthOut = sizeOut/sizeD;
	printf("Length: %d\n", lengthOut);
	printf("%.3f", out[16300/*131071*/]);
	printf("\n");
	cudaFree(d_in);
	cudaFree(d_out);
}

int main(int argc, char *argv[])
{
	{
		int noEls = 8;
		int noOfBlocks = 512;
		int r1 = 0;
		//int r2 = 1;
		int r2 = 1;
		int r3 = 2;
		int rplane = 3;
		int dim1 = 8;
		int dim2 = 16;
		int dim3 = 16;
		int dim4 = 32;
		int dim5 = 32;
		if(argc > 1)
		{
			noEls = atoi(argv[1]);
			noOfBlocks = atoi(argv[2]);
			r1 = atoi(argv[3]);
			r2 = atoi(argv[4]);
			r3 = atoi(argv[5]);
			rplane = atoi(argv[6]);
			dim1 = atoi(argv[7]);
			dim2 = atoi(argv[8]);
			dim3 = atoi(argv[9]);
			dim4 = atoi(argv[10]);
			dim5 = atoi(argv[11]);
		}
		run_test(noEls, noOfBlocks, r1, r2, r3, rplane, dim1, dim2, dim3, dim4, dim5);
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


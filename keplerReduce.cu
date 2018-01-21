
#include <time.h>
#include <stdio.h>
#include <limits.h>

void run_test();

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
int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
int blockReduceSum(int val) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
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

__global__ void deviceReduceKernel(int *in, int* out, int N) {
  float sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * 2* blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * 2 * gridDim.x) {
    sum += in[i] + in[i+blockDim.x];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

int main(void){

  {
    run_test();
  }

  
}

void run_test()
{
  const int N = 1 << 22;

  printf("N: %d\n", N);
  int *in, *out, *d_in, *d_out;

  in = (int*)malloc(N*sizeof(int));

  out = (int*)malloc(N*sizeof(int));

  for(int i=0; i<N; i++)
  {
    in[i] = 1;
  }

  cudaMalloc(&d_in, N*sizeof(int));
  cudaMalloc(&d_out, N*sizeof(int));

  //Event variables
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Transfer host data to device

  gpuErrchk(cudaMemcpy(d_in, in, N*sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_out, out, N*sizeof(int), cudaMemcpyHostToDevice));


  cudaEventRecord(start);
  deviceReduceKernel<<<64,256>>>(d_in,d_out, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms,start,stop);

  gpuErrchk(cudaMemcpy(out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost));

  double total = (N)*sizeof(int);

  double ebw = total/(ms*1e6);

  printf("EBW: %f\n", ebw);
  //Check errors
  

  for(int i=0;i<4;i++)
  {
    
    printf("%d %d ", out[i], i);
  }
}

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
int warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__inline__ __device__
int blockReduceSum(float val) {

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

__global__ void deviceReduceKernel(float *in, float* out, int N) {
  float sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
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
  const int N = 1 << 20;

  printf("N: %d\n", N);
  float *in, *out, *d_in, *d_out;

  in = (float*)malloc(N*sizeof(float));

  out = (float*)malloc(N*sizeof(float));

  for(int i=0; i<N; i++)
  {
    in[i] = 1/1000;
  }

  cudaMalloc(&d_in, N*sizeof(float));
  cudaMalloc(&d_out, N*sizeof(float));

  //Event variables
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Transfer host data to device

  gpuErrchk(cudaMemcpy(d_in, in, N*sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_out, out, N*sizeof(float), cudaMemcpyHostToDevice));


  cudaEventRecord(start);
  deviceReduceKernel<<<32,1024>>>(d_in,d_out, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms,start,stop);

  gpuErrchk(cudaMemcpy(out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost));

  double total = (N)*sizeof(float);

  double ebw = total/(ms*1e6);

  printf("EBW: %f\n", ebw);
  //Check errors
  

  for(int i=0;i<4;i++)
  {
    
    printf("%.3f %d ", out[i], i);
  }
}
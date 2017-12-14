/******************************************


 *******************************************/


#include <time.h>
#include <stdio.h>
#include <limits.h>

#define BLOCK_SIZE 32
void run_test();
//----------
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
// vector broadcast general except first dimension(x)

__global__ void broadcast_multi(float *x,float *y, float *z,
                                int *stride_x, int *stride_y,
                                int *stride_z,int N_z, int dimlen_z) {
    // ,int dimlen_z
    // const int dimlen_z=3;
    int index_z = threadIdx.x + (blockIdx.x * blockDim.x);
    int index_x,index_y;
    // int coords[dimlen_z];
    // int* coords = (int*)malloc(sizeof(int)*dimlen_z);
    int coords;

    int temp_index;
    while (index_z < N_z) {
        temp_index = index_z;
        index_x =0;
        index_y = 0;
        // n multiple of 2
        // (i/n) == (i>>log2(n))
        // we can place foo%n==foo&(n-1) if second dim is multiple of 2
        //try unsigned int

        for (int i=dimlen_z-1; i>0; i--)
        {
            coords = temp_index / (stride_z[i]);
            index_x+= stride_x[i]*coords;
            index_y+= stride_y[i]*coords;
            temp_index = temp_index %(stride_z[i]);
        }
            index_x+= temp_index;
            index_y+= temp_index;

        // for (int i=dimlen_z-1; i>=0; i--)
        // {
        //     coords = temp_index /(stride_z[i]);
        //     index_x+= stride_x[i]*coords;
        //     index_y+= stride_y[i]*coords;
        //     temp_index = temp_index &(stride_z[i]-1);
        // }
        // index_x =0;
        // index_y = 0;

        // z[index_z] = x[index_x]+y[index_y];
        z[index_z] = x[index_z]+y[index_z];

        index_z+=(blockDim.x * gridDim.x);

    }

}

int main(void)
{

    // int A_dimsizes[5] = {28,28,1344,4,4};
    // int A_len = 5;
    // int brdcastdim = 3;
    // for(int i = 0; i<10; i++)
    {
        run_test();
    }
    // run_test(100,10,1);

}


void run_test()
{
  // int stride_A[3] = {1,1,64};
  // // row stride
  // int stride_B[3] = {1,10,0};
  // int stride_Z[3] = {1,10,640};
  // int dimlen_z = 3;
  //
    const int A_len = 6;
    int A_dimsizes[A_len] = {50,50,20,40,4,4};
    int B_dimsizes[A_len] = {50,50,20,40,4,4};
    int A_strides[A_len];
    int A_N=1;

    A_strides[0]=1;
    int k=1;
    for (int i=0; i<A_len-1; i++)
    {
        k*=A_dimsizes[i];
        A_strides[i+1]=k;
        A_N*=A_dimsizes[i];
    }
    A_N*=A_dimsizes[A_len-1];


    // int B_dimsizes[A_len] = {28,28,1344,4,4};
    int B_len = A_len;
    int B_strides[B_len];
    int B_N=1;

    B_strides[0]=1;
    k=1;
    for (int i=0; i<B_len-1; i++)
    {
        k*=B_dimsizes[i];
        B_strides[i+1]=k;
        B_N*=B_dimsizes[i];
    }
    B_N*=B_dimsizes[A_len-1];

    for (int i=0; i<B_len; i++)
    {
        if (B_dimsizes[i]!=A_dimsizes[i])
        {
            if (B_dimsizes[i]==1)
            {
              B_strides[i]=0;
            }
            else{
              A_strides[i]=0;
            }

        }
    }
    int C_N = 1;
    int C_dimsizes[A_len];

    for (int i=0; i<B_len; i++)
    {

            if (B_dimsizes[i]>=A_dimsizes[i])
            {
                C_dimsizes[i]=B_dimsizes[i];
                C_N*=B_dimsizes[i];
            }
            else{
                C_dimsizes[i]=A_dimsizes[i];
                C_N*=A_dimsizes[i];
            }


    }

    int C_strides[A_len];
    C_strides[0]=1;
    k=1;
    for (int i=0; i<A_len-1; i++)
    {
        k*=C_dimsizes[i];
        C_strides[i+1]=k;
    }
    // C_N*=C_dimsizes[A_len-1];



    // we will broadcast to third dim (z)
    // // int brdcastdim = 3;
    // int brdcastdimstride = A_strides[brdcastdim-1];
    // int brdcastnextstride = A_strides[brdcastdim];
    // int multidimsize=1;

    // for (int i=brdcastdim; i<A_len; i++)
    // {
    //     multidimsize*=A_dimsizes[i];
    // }
    // int B_N=A_dimsizes[brdcastdim-1];

    float *A, *B,*C, *d_A, *d_B, *d_C;
    int *d_stride_A, *d_stride_B, *d_stride_C;
    A = (float*)malloc(A_N*sizeof(float));
    B = (float*)malloc(B_N*sizeof(float));
    C = (float*)malloc(C_N*sizeof(float));

    cudaMalloc(&d_A, A_N*sizeof(float));
    cudaMalloc(&d_B, B_N*sizeof(float));
    cudaMalloc(&d_C, C_N*sizeof(float));
    cudaMalloc(&d_stride_A, B_len*sizeof(int));
    cudaMalloc(&d_stride_B, B_len*sizeof(int));
    cudaMalloc(&d_stride_C, B_len*sizeof(int));
    // cudaMalloc(&d_coords, A_len*sizeof(int));

    for (int i = 0; i < A_N; i++) {
        A[i] = 1;

    }
    for (int i = 0; i < B_N; i++) {
        B[i] = 2;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    gpuErrchk(cudaMemcpy(d_A, A, A_N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, B_N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_C, C, C_N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_stride_A, A_strides, A_len*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_stride_B, B_strides, A_len*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_stride_C, C_strides, A_len*sizeof(int), cudaMemcpyHostToDevice));

    cudaEventRecord(start);


    int ITER=10;
    for (int i=0; i<ITER; i++)
    {

      broadcast_multi<<<256,256>>>(d_A,d_B, d_C,
       d_stride_A, d_stride_B,
       d_stride_C,C_N,A_len);

    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds = milliseconds/ITER;


    gpuErrchk(cudaMemcpy(C, d_C, C_N*sizeof(float), cudaMemcpyDeviceToHost));


    int maxError = 0;

    for (int i = 0; i < C_N; i++)
    {
          if(C[i]!=3)
          {
              maxError+=1;
              printf("error value %d index  %d \n",C[i],i);
          }
    }

    for (int i = 0; i < A_len; i++)
    {
              printf("Astride %d : %d \n",i,A_strides[i]);
    }
    for (int i = 0; i < A_len; i++)
    {
              printf("Bstride %d : %d \n",i,B_strides[i]);
    }
    for (int i = 0; i < A_len; i++)
    {
              printf("Cstride %d : %d \n",i,C_strides[i]);
    }
    for (int i = 0; i < A_len; i++)
    {
              printf("C_dimsizes %d : %d \n",i,C_dimsizes[i]);
    }



    if (maxError!=0)
    {
        printf("Max error: %d\n", maxError);
    }
    double total_data = (((double)A_N)+(double)B_N+(double)C_N)*4;
    double gflop = (A_N)/(milliseconds*1e6);
    // printf("total data: %f\n size1 %f\n size2\n", total_data,size1);
    // printf("milliseconds: %f\n", milliseconds);
    //gflop
    // printf("%f\n", gflop);
    // Effective Bandwidth
    printf("Effective Bandwidth %f\n", total_data/milliseconds/1e6);
}
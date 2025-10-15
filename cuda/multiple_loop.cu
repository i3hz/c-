#include <stdio.h>

__global__ void loop(int N)
{
  int i  = threadIdx.x + blockDim.x * blockIdx.x;
  printf("This is iteration number %d\n", i);
  
}

int main()
{
  loop<<<2,5>>>(10);
  cudaDeviceSynchronize();
}

#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int device;
    cudaDeviceProp props;
    
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    printf("Compute capability of device %d: %d.%d\n", device, props.major, props.minor);

    return 0;
}

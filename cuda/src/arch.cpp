#include <iostream>

using namespace std;

__global__ void kernel_max_pooling(int padding, float *in, float *out)
{
}

int main(int argc, char *argv[])
{
    if (argv[1] == NULL)
    {
        cout << "Missing an argument, Please input a padding size" << endl;
        exit(-1);
    }
    int padding = atoi(argv[1]);
    cout << "Program start, padding is: " << padding << endl;

    // TODO: test suits

    float *host_in, *host_out;
    float *dev_in, *dev_out;
    // allocate memory for host
    host_in = (float *)malloc(sizeof(float)); // TODO: allocate the size of array (an image)
    if (host_in == NULL)
    {
        cout << "Host Memory allocation fails" << endl;
        exit(-1);
    }
    host_out = (float *)malloc(sizeof(float)); // TODO: allocate the size of array (an image)
    if (host_out == NULL)
    {
        cout << "Host Memory allocation fails" << endl;
        exit(-1);
    }

    // create buffer on device
    cudaError_t err = cudaMalloc(&dev_in, ); // TODO: allocate the size of array (an image)
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    err = cudaMalloc(&dev_out, ); // TODO: allocate the size of array (an image)
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    // copy data into device
    cudaMemcpy(dev_in, host_in, , cudaMemcpyHostToDevice); // TODO: pass the size of array into to function

    // create GPU timing events for timing the GPU
    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    // calling maxpooling kernel
    cudaEventRecord(st2);
    kernel_max_pooling<<<>>>(padding, dev_in, dev_out);
    cudaEventRecord(et2);

    // host waits until et2 has occured
    cudaEventSynchronize(et2);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st2, et2);
    cout << "Kernel time: " << milliseconds << "ms" << endl;
    // copy data out
    cudaMemcpy(host_out, dev_out, , cudaMemcpyDeviceToHost); // TODO: pass the size of array into to function

    // free memory
    free(host_in);
    free(host_out);
    cudaFree(dev_in);
    cudaFree(dev_out);
    cout << "Program end" << endl;
}
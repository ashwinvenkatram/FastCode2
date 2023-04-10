#include <iostream>
#include <float.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>


using namespace std;

unsigned long long NUM_RUNS = 1;

__global__ void kernel_max_pooling(int padding, int input_size, int input_width, int input_height, int output_size, int output_width, float *in, float *out)
{
        // Determine the thread and block index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Iterate over the input image or feature map
    for (int idx = tid; idx < input_size; idx += stride)
    {
        // Calculate the row and column index
        int row = idx / input_width;
        int col = idx % input_width;

        // Calculate the output index
        int out_idx = (row - padding) * output_width + (col - padding);

        // Initialize the max value to the minimum float value
        float max_val = -FLT_MAX;

        // Iterate over the pooling window
        for (int i = -padding; i <= padding; i++)
        {
            for (int j = -padding; j <= padding; j++)
            {
                // Calculate the input index
                int input_row = row + i;
                int input_col = col + j;
                int input_idx = input_row * input_width + input_col;

                // Check if the input index is valid
                if (input_row >= 0 && input_row < input_height &&
                    input_col >= 0 && input_col < input_width)
                {
                    // Update the max value if necessary
                    float val = in[input_idx];
                    if (val > max_val)
                    {
                        max_val = val;
                    }
                }
            }
        }

        // Write the max value to the output
        out[out_idx] = max_val;
    }
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

    size_t N = 64;
    float sum = 0.0;
    float sum1 = 0.0;
    float *rdtsc_arr;

    rdtsc_arr = (float *)calloc(NUM_RUNS, sizeof(float));

    double average, variance, std_dev;

    for (unsigned long long run_id = 0; run_id < NUM_RUNS; run_id++)
    {
        // create buffer on host
        host_in = (float *)malloc(N * N * sizeof(float));
        host_out = (float *)malloc(N * N * sizeof(float));

        // creates a matrix stored in row major order
        for (int i = 0; i != N; ++i)
        {
            for (int j = 0; j != N; ++j)
            {
                host_in[i * N + j] = i * N + j;
            }
        }

        // allocate memory for device
        cudaError_t err = cudaMalloc(&dev_in, N * N * sizeof(float)); // TODO: allocate the size of array (an image)
        if (err != cudaSuccess)
        {
            cout << "Dev Memory not allocated" << endl;
            exit(-1);
        }

        err = cudaMalloc(&dev_out, N * N * sizeof(float)); // TODO: allocate the size of array (an image)
        if (err != cudaSuccess)
        {
            cout << "Dev Memory not allocated" << endl;
            exit(-1);
        }

        // copy data into device
        cudaMemcpy(dev_in, host_in, N * N * sizeof(float), cudaMemcpyHostToDevice); // TODO: pass the size of array into to function

        // create GPU timing events for timing the GPU
        cudaEvent_t st2, et2;
        cudaEventCreate(&st2);
        cudaEventCreate(&et2);

        // calling maxpooling kernel
        cudaEventRecord(st2);
        dim3 grid(1);
        dim3 block(128);
        // int padding, float input_size, float input_width, float input_height, float output_size, float output_width, float *in, float *out)

        kernel_max_pooling<<<grid, block>>>(padding, N * N, N, N, N * N, N, dev_in, dev_out);
        cudaEventRecord(et2);

        // host waits until et2 has occured
        cudaEventSynchronize(et2);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, st2, et2);
        cout << "Kernel time: " << milliseconds << "ms" << endl;
        // copy data out
        cudaMemcpy(host_out, dev_out, N * N * sizeof(float), cudaMemcpyDeviceToHost); // TODO: pass the size of array into to function

        // free memory
        free(host_in);
        free(host_out);
        cudaFree(dev_in);
        cudaFree(dev_out);
    }

   
}
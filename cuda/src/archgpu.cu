#include <iostream>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "../include/utils.h"
// #include "kernel_configs.h"

using namespace std;

unsigned long long NUM_RUNS = 1;

// __global__ void kernel_max_pooling(int padding, int input_size, int input_width, int input_height, int output_size, int output_width, float *in, float *out)
__global__ void kernel_max_pooling(FLOATTYPE *in, FLOATTYPE *out, int c, 
                                    int i_h, int i_w, int input_size,
                                    int f_w, int f_h, int o_h, int o_w, int padding)
{
    // Determine the thread and block index
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ashwin: Outer loop un-necessary (?) tid should implicitly account for the centre of the receptive field
    // Iterate over the input image or feature map
    // for (int idx = tid; idx < input_size; idx += stride)
    
    {
        // Calculate the row and column index
        int row = tidx / i_w;
        int col = tidx % i_h;

        // Calculate the output index
        // int out_idx = (row - padding) * output_width + (col - padding);

        // Initialize the max value to the minimum float value
        FLOATTYPE max_val = -FLT_MAX;

        // Iterate over the pooling window
        for (int i = -f_w / 2; i <= f_w / 2; i++) {
            for (int j = -f_h / 2; j <= f_h / 2; j++) {
                // Calculate the input index, accounting for (left side) padding
                // right side padding is inconsequential
                int input_row = row + i;
                int input_col = col + j;

                int input_idx = input_row * i_w + input_col;

                // Check if the input index is valid
                // clamp the index to the range of the input matrix
                if (input_row >= 0 && input_row < i_h &&
                    input_col >= 0 && input_col < i_w)
                {
                    // Update the max value if necessary
                    FLOATTYPE val = in[input_idx];
                    max_val = fmaxf(max_val, val);
                }
            }
        }

        // Write the max value to the output
        int out_idx = row * o_w + col;
        out[out_idx] = max_val;
    }
}


__global__ void cudaMaxPool_ref(FLOATTYPE *gOutImage, FLOATTYPE *gImage, int c, int h, int w, int fw, int fh)
{
    // Tile size
    int tw = blockDim.x;
    int th = blockDim.y;

    // Padded tile size
    int pTW = tw + fw - 1;
    int pTH = th + fh - 1;

    extern __shared__ FLOATTYPE shmem[];

    // Tile offsets in image. Without Padding
    int tileWidthOffset = tw * blockIdx.x;
    int tileHeightOffset = th * blockIdx.y;
    int channel = blockIdx.z;

    // Creating input data with padding
    for (int x = threadIdx.x; x < pTW; x += tw)
    {
        int copy_x = x - fw / 2 + tileWidthOffset;
        for (int y = threadIdx.y; y < pTH; y += tw)
        {
            int copy_y = y - fh / 2 + tileHeightOffset;

            int shmem_idx = shmem_offset(0, 0, x, y, pTW, 0, 0);

            if (copy_x < 0 || copy_x >= w || copy_y < 0 || copy_y >= h)
            {
                shmem[shmem_idx] = 0;
            }
            else
            {
                shmem[shmem_idx] = gImage[indexToOffset(copy_x, copy_y, channel, h, w, 0, 0)];
            }
        }
    }

    __syncthreads();

    // Pixel this thread is responsible for
    int widthOffset = tileWidthOffset + threadIdx.x;
    int heightOffset = tileHeightOffset + threadIdx.y;

    if (widthOffset < 0 || widthOffset >= w || heightOffset < 0 || heightOffset >= h)
    {
        return;
    }
    
    FLOATTYPE maxValue = shmem[shmem_offset(threadIdx.x, threadIdx.y, 0, 0, pTW, fw / 2, fh / 2)];
    for (int x = -fw / 2; x <= fw / 2; x++)
    {
        for (int y = -fh / 2; y <= fh / 2; y++)
        {
            FLOATTYPE value = shmem[shmem_offset(x, y, threadIdx.x, threadIdx.y, pTW, fw / 2, fh / 2)];
            if (value > maxValue)
            {
                maxValue = value;
            }
        }
    }

    gOutImage[indexToOffset(0, 0, channel, h, w, heightOffset, widthOffset)] = maxValue;
}


void cudaMaxPooling(int padding, int c, int i_h, int i_w, int f_dim)
{   
    int fw = f_dim;
    int fh = f_dim;

    int input_size = i_h * i_w;
    long int imageSize = sizeof(FLOATTYPE) * c * input_size;

    FLOATTYPE *cImage = (FLOATTYPE *)malloc(imageSize);
    FLOATTYPE *gImage;

    // TODO: Stride is 1 for the workload; change if required (from refgpu.cu)
    // output spatial dimensions; setting strides to 1
    int s_h = 1;
    int s_w = 1;
    int p_h = padding;
    int p_w = padding;

    int o_h = ((i_h + 2 * p_h - fh) / s_h) + 1;
    int o_w = ((i_w + 2 * p_w - fw) / s_w) + 1;

    printf("output dims: %d, %d\n", o_h, o_w);
    
    long int outImageSize = sizeof(FLOATTYPE) * c * o_w * o_h;

    FLOATTYPE *cOutImage = (FLOATTYPE *)malloc(outImageSize);
    FLOATTYPE *gOutImage;
    
    struct timespec start, end;

    CUDA_CALL(cudaMalloc((void **)&gImage, imageSize));
    CUDA_CALL(cudaMalloc((void **)&gOutImage, outImageSize));

    CUDA_CALL(cudaMemset((void *)gOutImage, 0, outImageSize));

    // Does not include padding
    fillImage_floattype(cImage, c, i_h, i_w);

    printf("I = checksum: %lf\n", calculateChecksum_float(cImage, c, i_h, i_w));
    
    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaMemcpy(gImage, cImage, imageSize, cudaMemcpyHostToDevice));
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    printf("Copy host->dev %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

    int shmem_size = sizeof(FLOATTYPE) * (TW + fw - 1) * (TH + fh - 1);
    // dim3 blockDim(TW, TH);
    // dim3 gridDim(DIV_RUP(i_w, TW), DIV_RUP(i_h, TH), c);
    dim3 gridDim(1);
    dim3 blockDim(192);

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    kernel_max_pooling<<<gridDim, blockDim, shmem_size>>>(gImage, gOutImage, c, i_h, i_w, input_size, 
                                                            fw, fh, o_h, o_w, padding);

    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaGetLastError());
    printf("Time cuda code %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    printf("Copy dev->host %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

    printf("CUDA O = checksum: %f\n", calculateChecksum_float(cOutImage, c, o_h, o_w));

    printf("Input tensor:\n");
    print_tensor(cImage, c, i_h, i_w);
    printf("Output tensor:\n");
    print_tensor(cOutImage, c, o_h, o_w);
    
    free(cImage);
    free(cOutImage);
    CUDA_CALL(cudaFree(gImage));
    CUDA_CALL(cudaFree(gOutImage));
}


void cudaMaxPooling_Ref(int c, int h, int w, int f_dim)
{
    int fw = f_dim;
    int fh = f_dim;

    long int imageSize = sizeof(FLOATTYPE) * c * w * h;

    FLOATTYPE *cImage = (FLOATTYPE *)malloc(imageSize);
    FLOATTYPE *gImage;

    // TODO: Stride is 1 for the workload; change if required (from refgpu.cu)
    long int outImageSize = sizeof(FLOATTYPE) * c * w * h;

    FLOATTYPE *cOutImage = (FLOATTYPE *)malloc(outImageSize);
    FLOATTYPE *gOutImage;

    struct timespec start, end;

    CUDA_CALL(cudaMalloc((void **)&gImage, imageSize));
    CUDA_CALL(cudaMalloc((void **)&gOutImage, outImageSize));

    CUDA_CALL(cudaMemset((void *)gOutImage, 0, outImageSize));

    // Does not include padding
    fillImage_floattype(cImage, c, h, w);

    printf("I = checksum: %lf\n", calculateChecksum_float(cImage, c, h, w));

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    CUDA_CALL(cudaMemcpy(gImage, cImage, imageSize, cudaMemcpyHostToDevice));
    
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    printf("Copy host->dev %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

    int shmem_size = sizeof(FLOATTYPE) * (TW + fw - 1) * (TH + fh - 1);
    dim3 blockDim(TW, TH);
    dim3 gridDim(DIV_RUP(w, TW), DIV_RUP(h, TH), c);

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    cudaMaxPool_ref<<<gridDim, blockDim, shmem_size>>>(gOutImage, gImage, c, h, w, fw, fh);

    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaGetLastError());
    printf("Time cuda code %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    printf("Copy dev->host %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

    printf("CUDA O = checksum: %f\n", calculateChecksum_float(cOutImage, c, h, w));

    printf("Input tensor:\n");
    print_tensor(cImage, c, h, w);
    printf("Output tensor:\n");
    print_tensor(cOutImage, c, h, w);

    free(cImage);
    free(cOutImage);
    CUDA_CALL(cudaFree(gImage));
    CUDA_CALL(cudaFree(gOutImage));
}


// int main(int argc, char *argv[])
// {
//     // if (argv[1] == NULL)
//     // {
//     //     cout << "Missing an argument, Please input a padding size" << endl;
//     //     exit(-1);
//     // }
//     // int padding = atoi(argv[1]);
//     // cout << "Program start, padding is: " << padding << endl;
//     // TODO: test suits
//     FLOATTYPE *host_in, *host_out;
//     FLOATTYPE *dev_in, *dev_out;
//     size_t N = 64;
//     float sum = 0.0;
//     float sum1 = 0.0;
//     float *rdtsc_arr;
//     rdtsc_arr = (float *)calloc(NUM_RUNS, sizeof(float));
//     double average, variance, std_dev;
//     for (unsigned long long run_id = 0; run_id < NUM_RUNS; run_id++)
//     {
//         // create buffer on host
//         host_in = (FLOATTYPE *)malloc(N * N * sizeof(FLOATTYPE));
//         host_out = (FLOATTYPE *)malloc(N * N * sizeof(FLOATTYPE));
//         // creates a matrix stored in row major order
//         for (int i = 0; i != N; ++i)
//         {
//             for (int j = 0; j != N; ++j)
//             {
//                 host_in[i * N + j] = i * N + j;
//             }
//         }
//         // allocate memory for device
//         cudaError_t err = cudaMalloc(&dev_in, N * N * sizeof(FLOATTYPE)); // TODO: allocate the size of array (an image)
//         if (err != cudaSuccess)
//         {
//             cout << "Dev Memory not allocated" << endl;
//             exit(-1);
//         }
//         err = cudaMalloc(&dev_out, N * N * sizeof(FLOATTYPE)); // TODO: allocate the size of array (an image)
//         if (err != cudaSuccess)
//         {
//             cout << "Dev Memory not allocated" << endl;
//             exit(-1);
//         }
//         // copy data into device
//         cudaMemcpy(dev_in, host_in, N * N * sizeof(FLOATTYPE), cudaMemcpyHostToDevice); // TODO: pass the size of array into to function
//         // create GPU timing events for timing the GPU
//         cudaEvent_t st2, et2;
//         cudaEventCreate(&st2);
//         cudaEventCreate(&et2);
//         // calling maxpooling kernel
//         cudaEventRecord(st2);
//         dim3 grid(1);
//         dim3 block(128);
//         // int padding, float input_size, float input_width, float input_height, float output_size, float output_width, float *in, float *out)
//         kernel_max_pooling<<<grid, block>>>(padding, N * N, N, N, N * N, N, dev_in, dev_out);
//         cudaEventRecord(et2);
//         // host waits until et2 has occured
//         cudaEventSynchronize(et2);
//         float milliseconds = 0;
//         cudaEventElapsedTime(&milliseconds, st2, et2);
//         cout << "Kernel time: " << milliseconds << "ms" << endl;
//         // copy data out
//         cudaMemcpy(host_out, dev_out, N * N * sizeof(FLOATTYPE), cudaMemcpyDeviceToHost); // TODO: pass the size of array into to function
//         // free memory
//         free(host_in);
//         free(host_out);
//         cudaFree(dev_in);
//         cudaFree(dev_out);
//     }
// }

int main(int argc, char *argv[])
{
    int C = atoi(argv[1]);
    int H = atoi(argv[2]);
    int W = atoi(argv[3]);
    // Assume that filter dims is square FW = FH
    int F_dim = atoi(argv[4]);
    
    // padding is always floor(F_dim/ 2) to ensure tensor output dims == tensor input dims
    int padding = F_dim/ 2;
    
    printf("C:%d; H:%d; W:%d; F_dim:%d; padding:%d\n", C, H, W, F_dim, padding);
    printf("Reference Max Pool Using CUDA\n");
    // internally handles padding logic
    cudaMaxPooling_Ref(C, H, W, F_dim);
    printf("\n");

    printf("FC2: Max Pool Using CUDA\n");
    // requires padding as input & perform correction
    // (int padding, int c, int i_h, int i_w, int fw, int fh)
    cudaMaxPooling(padding, C, H, W, F_dim);

    return 0;
}
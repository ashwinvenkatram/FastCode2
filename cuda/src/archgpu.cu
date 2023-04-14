#include <iostream>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "../include/utils.h"

using namespace std;

__global__ void kernel_max_pooling(FLOATTYPE *in, FLOATTYPE *out, int c, 
                                    int i_h, int i_w, int input_spatial_size,
                                    int f_w, int f_h, int o_h, int o_w, int output_spatial_size, int padding)
{
    // Determine the thread and block index
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    // Outer loop un-necessary; tidx implicitly account for the centre of the receptive field
    
    // hack for now; TODO: Change launch configs to handle this better
    if(tidx < 169)
    {
        // Iterate over the input channel dim
        for (int c_iter = 0; c_iter < c; c_iter++){

            // Calculate the spatial row and column index
            int row = tidx / i_w;
            int col = tidx % i_h;

            // Initialize the max value to the minimum float value
            FLOATTYPE max_val = 0.0;

            int input_channel_offset = c_iter * input_spatial_size;
            int output_channel_offset = c_iter * output_spatial_size;

            // Iterate over the pooling window
            for (int i = -f_w / 2; i <= f_w / 2; i++) {
                for (int j = -f_h / 2; j <= f_h / 2; j++) {
                    // Calculate the input index, accounting for (left side) padding
                    // right side padding is inconsequential
                    int input_row = row + i;
                    int input_col = col + j;
                    
                    // Check if the input index is valid to sub-tensor
                    // clamp the index to the range of the input matrix
                    // else encroaches into previous tensor
                    if (input_row >= 0 && input_row < i_h &&
                        input_col >= 0 && input_col < i_w)
                    {
                        int input_idx = input_channel_offset + input_row * i_w + input_col;
                        // Update the max value if necessary
                        FLOATTYPE val = in[input_idx];
                        max_val = fmaxf(max_val, val);
                    }
                }
            }

            // Write the max value to the output
            int out_idx = output_channel_offset + row * o_w + col;
            out[out_idx] = max_val;
        }
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


FLOATTYPE cudaMaxPooling(int padding, int c, int i_h, int i_w, int f_dim)
{   
    int fw = f_dim;
    int fh = f_dim;

    int input_spatial_size = i_h * i_w;
    long int imageSize = sizeof(FLOATTYPE) * c * input_spatial_size;

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
    
    int output_spatial_size = o_w * o_h;
    long int outImageSize = sizeof(FLOATTYPE) * c * output_spatial_size;
    
    FLOATTYPE *cOutImage = (FLOATTYPE *)malloc(outImageSize);
    FLOATTYPE *gOutImage;
    
    struct timespec start, end;

    CUDA_CALL(cudaMalloc((void **)&gImage, imageSize));
    CUDA_CALL(cudaMalloc((void **)&gOutImage, outImageSize));

    CUDA_CALL(cudaMemset((void *)gOutImage, 0, outImageSize));

    // Does not include padding
    fillImage_floattype(cImage, c, i_h, i_w);

    FLOATTYPE input_checksum = calculateChecksum_float(cImage, c, i_h, i_w);
    printf("I = checksum: %f\n", input_checksum);
    
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
    
    kernel_max_pooling<<<gridDim, blockDim, shmem_size>>>(gImage, gOutImage, c, i_h, i_w, input_spatial_size, 
                                                            fw, fh, o_h, o_w, output_spatial_size, padding);

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

    FLOATTYPE output_checksum = calculateChecksum_float(cOutImage, c, o_h, o_w);
    printf("CUDA O = checksum: %f\n", output_checksum);

    #ifdef PRINT_DEBUG
        printf("Input tensor:\n");
        print_tensor(cImage, c, i_h, i_w);
        printf("Output tensor:\n");
        print_tensor(cOutImage, c, o_h, o_w);
    #endif

    free(cImage);
    free(cOutImage);
    CUDA_CALL(cudaFree(gImage));
    CUDA_CALL(cudaFree(gOutImage));

    return output_checksum;
}


FLOATTYPE cudaMaxPooling_Ref(int c, int h, int w, int f_dim)
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

    FLOATTYPE input_checksum = calculateChecksum_float(cImage, c, h, w);
    printf("I = checksum: %f\n", input_checksum);

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

    FLOATTYPE output_checksum = calculateChecksum_float(cOutImage, c, h, w);
    printf("CUDA O = checksum: %f\n", output_checksum);

    #ifdef PRINT_DEBUG
        printf("Input tensor:\n");
        print_tensor(cImage, c, h, w);
        printf("Output tensor:\n");
        print_tensor(cOutImage, c, h, w);
    #endif

    free(cImage);
    free(cOutImage);
    CUDA_CALL(cudaFree(gImage));
    CUDA_CALL(cudaFree(gOutImage));

    return output_checksum;
}

int main(int argc, char *argv[])
{
    int C = atoi(argv[1]);
    int H = atoi(argv[2]);
    int W = atoi(argv[3]);
    // Assume that filter dims is square FW = FH
    int F_dim = atoi(argv[4]);

    int NUM_RUNS = atoi(argv[5]);
    
    // padding is always floor(F_dim/ 2) to ensure tensor output dims == tensor input dims
    int padding = F_dim/ 2;
    
    for(int run=0; run < NUM_RUNS; run++){
        printf("C:%d; H:%d; W:%d; F_dim:%d; padding:%d\n", C, H, W, F_dim, padding);
        printf("Reference Max Pool Using CUDA\n");
        // internally handles padding logic
        FLOATTYPE ref_checksum = cudaMaxPooling_Ref(C, H, W, F_dim);
        printf("\n");

        printf("FC2: Max Pool Using CUDA\n");
        // requires padding as input & perform correction
        // (int padding, int c, int i_h, int i_w, int fw, int fh)
        FLOATTYPE kernel_checksum = cudaMaxPooling(padding, C, H, W, F_dim);

        assert(ref_checksum == kernel_checksum);
        printf("==============================\n\n");
    }
    return 0;
}
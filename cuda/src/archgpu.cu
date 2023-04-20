#include <iostream>
#include <float.h>
#include <cudnn.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "../include/utils.h"

using namespace std;

// #define PRINT_PER_RUN 1

__global__ void integrated_kernel_max_pooling(FLOATTYPE *in, FLOATTYPE *out0, FLOATTYPE *out1, FLOATTYPE *out2, int c, 
                                    int i_h, int i_w, int input_spatial_size,
                                    int f_w0, int f_h0, int f_w1, int f_h1, int f_w2, int f_h2, int o_h, int o_w, int output_spatial_size)
{
    // 0: f_dim 5
    // 1: f_dim 9
    // 2: f_dim 13

    // Spatial access index
    int tidx = threadIdx.x;
    // Channel dim access
    int c_iter = blockIdx.x;
    
    // Calculate the spatial row and column index
    // invariant to channel iteration
    int row = tidx / i_w;
    int col = tidx % i_h;

    // if thread is accessing out-of-bounds, return
    if (row < 0 || row >= i_h || col < 0 || col >= i_w)
    {
        return; //make it compute redundant
    }

    // Declare shared memory
    extern __shared__ FLOATTYPE smem[];

    int input_channel_offset = c_iter * input_spatial_size;
    int output_channel_offset = c_iter * output_spatial_size;

    // Calculate the input index
    int input_idx = input_channel_offset + row * i_w + col;

    // Load input data into shared memory
    smem[tidx] = in[input_idx];
    __syncthreads();

    // Initialize the max value to the minimum float value
    FLOATTYPE max_val0 = -FLT_MAX;
    FLOATTYPE max_val1 = -FLT_MAX;
    FLOATTYPE max_val2 = -FLT_MAX;

    // Iterate over the pooling window
    for (int i = -f_w2 / 2; i <= f_w2 / 2; i++) {
        for (int j = -f_h2 / 2; j <= f_h2 / 2; j++) {
            // Calculate the shmem access index
            int smem_row = row + i;
            int smem_col = col + j;
            int smem_idx = smem_row * i_w + smem_col;

            // Check if the shared memory index is valid
            if (smem_row >= 0 && smem_row < i_h &&
                smem_col >= 0 && smem_col < i_w)
            {
                // Update the max value if necessary
                FLOATTYPE val = smem[smem_idx];
                max_val2 = fmaxf(max_val2, val);
                
                if(i <=f_w0/2 && j <=f_h0/2){
                    // valid to update max_val0, max_val1
                    max_val0 = fmaxf(max_val0, val);
                    max_val1 = fmaxf(max_val1, val);
                }
                else if(i <=f_w1/2 && j <=f_h1/2){
                    // valid to update max_val1
                    max_val1 = fmaxf(max_val1, val);
                }
            }
        }
    }

    // Write the max value to the output
    int out_idx = output_channel_offset + row * o_w + col;
    out0[out_idx] = max_val0;
    out1[out_idx] = max_val1;
    out2[out_idx] = max_val2;
}


__global__ void kernel_max_pooling(FLOATTYPE *in, FLOATTYPE *out, int c, 
                                    int i_h, int i_w, int input_spatial_size,
                                    int f_w, int f_h, int o_h, int o_w, int output_spatial_size)
{
    // Spatial access index
    int tidx = threadIdx.x;
    // Channel dim access
    int c_iter = blockIdx.x;
    
    // Calculate the spatial row and column index
    // invariant to channel iteration
    int row = tidx / i_w;
    int col = tidx % i_h;

    // if thread is accessing out-of-bounds, return
    if (row < 0 || row >= i_h || col < 0 || col >= i_w)
    {
        return;
    }

    // Declare shared memory
    extern __shared__ FLOATTYPE smem[];

    int input_channel_offset = c_iter * input_spatial_size;
    int output_channel_offset = c_iter * output_spatial_size;

    // Calculate the input index
    int input_idx = input_channel_offset + row * i_w + col;

    // Load input data into shared memory
    smem[tidx] = in[input_idx];
    __syncthreads();

    // Initialize the max value to the minimum float value
    FLOATTYPE max_val = -FLT_MAX;

    // Iterate over the pooling window
    for (int i = -f_w / 2; i <= f_w / 2; i++) {
        for (int j = -f_h / 2; j <= f_h / 2; j++) {
            // Calculate the shared memory index
            int smem_row = row + i;
            int smem_col = col + j;
            int smem_idx = smem_row * i_w + smem_col;
            
            // Check if the shared memory index is valid
            if (smem_row >= 0 && smem_row < i_h &&
                smem_col >= 0 && smem_col < i_w)
            {
                // Update the max value if necessary
                FLOATTYPE val = smem[smem_idx];
                max_val = fmaxf(max_val, val);
            }
        }
    }

    // Write the max value to the output
    int out_idx = output_channel_offset + row * o_w + col;
    out[out_idx] = max_val;
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

/**
 * Designed for integrated 3 max-pool kernels together
*/
FLOATTYPE cudaMaxPoolingIntegrated(int c, int i_h, int i_w, int* f_dim, double &sum, double &H2D_time, double &D2H_time, double *timing_arr)
{   
    double kernel_time = 0.0;
    int NUM_KERNELS = 3;

    int fw0 = f_dim[0];
    int fh0 = f_dim[0];

    int fw1 = f_dim[1];
    int fh1 = f_dim[1];

    int fw2 = f_dim[2];
    int fh2 = f_dim[2];

    int input_spatial_size = i_h * i_w;
    long int imageSize = sizeof(FLOATTYPE) * c * input_spatial_size;

    FLOATTYPE *cImage = (FLOATTYPE *)malloc(imageSize);
    FLOATTYPE *gImage;

    // TODO: Stride is 1 for the workload; change if required (from refgpu.cu)
    // output spatial dimensions; setting strides to 1
    int s_h = 1;
    int s_w = 1;
    
    // padding is always floor(F_dim/ 2) to ensure tensor output dims == tensor input dims
    int p_h0 = fh0/ 2;
    int p_w0 = fw0/ 2;
    
    int p_h1 = fh1/ 2;
    int p_w1 = fw1/ 2;
    
    int p_h2 = fh2/ 2;
    int p_w2 = fw2/ 2;

    int o_h0 = ((i_h + 2 * p_h0 - fh0) / s_h) + 1;
    int o_w0 = ((i_w + 2 * p_w0 - fw0) / s_w) + 1;

    int o_h1 = ((i_h + 2 * p_h1 - fh1) / s_h) + 1;
    int o_w1 = ((i_w + 2 * p_w1 - fw1) / s_w) + 1;

    int o_h2 = ((i_h + 2 * p_h2 - fh2) / s_h) + 1;
    int o_w2 = ((i_w + 2 * p_w2 - fw2) / s_w) + 1;

    // ensure that all output spatial dimensions are equal
    // Requirement for kernel design
    assert(o_h0==o_h1 && o_h1==o_h2);
    assert(o_w0==o_w1 && o_w1==o_w2);

    int output_spatial_size = o_w0 * o_h0;
    long int outImageSize = sizeof(FLOATTYPE) * NUM_KERNELS * c * output_spatial_size;
    FLOATTYPE *cOutImage = (FLOATTYPE *)malloc(outImageSize);
    FLOATTYPE *gOutImage;
    
    struct timespec start, end;

    CUDA_CALL(cudaMalloc((void **)&gImage, imageSize));
    CUDA_CALL(cudaMalloc((void **)&gOutImage, outImageSize));

    CUDA_CALL(cudaMemset((void *)gOutImage, 0, outImageSize));

    // Maintaining output tensor pointer to single malloc'ed block
    FLOATTYPE *gOutImage0 = gOutImage;
    FLOATTYPE *gOutImage1 = gOutImage + (c * output_spatial_size);
    FLOATTYPE *gOutImage2 = gOutImage1 + (c * output_spatial_size);

    // Does not include padding
    fillImage_floattype(cImage, c, i_h, i_w);

    FLOATTYPE input_checksum = calculateChecksum_float(cImage, c, i_h, i_w);
    
    #ifdef PRINT_PER_RUN
        printf("I = checksum: %f\n", input_checksum);
    #endif

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaMemcpy(gImage, cImage, imageSize, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    H2D_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    
    #ifdef PRINT_PER_RUN
        printf("Copy host->dev %lf sec\n", H2D_time);
    #endif

    int shmem_size = sizeof(FLOATTYPE) * input_spatial_size;

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    // kernel_max_pooling<<<c, output_spatial_size, shmem_size>>>(gImage, gOutImage0, c, i_h, i_w, input_spatial_size, 
    //                                                         fw0, fh0, o_h0, o_w0, output_spatial_size);
    // kernel_max_pooling<<<c, output_spatial_size, shmem_size>>>(gImage, gOutImage1, c, i_h, i_w, input_spatial_size, 
    //                                                         fw1, fh1, o_h0, o_w0, output_spatial_size);
    // kernel_max_pooling<<<c, output_spatial_size, shmem_size>>>(gImage, gOutImage2, c, i_h, i_w, input_spatial_size, 
    //                                                         fw2, fh2, o_h0, o_w0, output_spatial_size);

    integrated_kernel_max_pooling<<<c, output_spatial_size, shmem_size>>>(gImage, gOutImage0, gOutImage1, gOutImage2, c, i_h, i_w, input_spatial_size, 
                                                            fw0, fh0, fw1, fh1, fw2, fh2, o_h0, o_w0, output_spatial_size);
    
    cudaDeviceSynchronize();                                          
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    CUDA_CALL(cudaGetLastError());
    kernel_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);

    #ifdef PRINT_PER_RUN
        printf("Time cuda code %lf sec\n", kernel_time);
    #endif

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    D2H_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    #ifdef PRINT_PER_RUN
        printf("Copy dev->host %lf sec\n",D2H_time);
    #endif

    // Verify results for each subtensor
    FLOATTYPE output_checksum0 = calculateChecksum_float(cOutImage, c, o_h0, o_w0);
    FLOATTYPE output_checksum1 = calculateChecksum_float(cOutImage + (c * output_spatial_size), c, o_h0, o_w0);
    FLOATTYPE output_checksum2 = calculateChecksum_float(cOutImage + 2*(c * output_spatial_size), c, o_h0, o_w0);

    #ifdef PRINT_PER_RUN
        printf("CUDA O = checksum: %f\n", output_checksum0);
        printf("CUDA O = checksum: %f\n", output_checksum1);
        printf("CUDA O = checksum: %f\n", output_checksum2);
    #endif

    FLOATTYPE output_checksum = output_checksum0 + output_checksum1 + output_checksum2;

    #ifdef PRINT_DEBUG
        printf("Input tensor:\n");
        print_tensor(cImage, c, i_h, i_w);
        printf("Output tensor:\n");
        print_tensor(cOutImage, c, o_h, o_w);
    #endif

    // update timing
    sum += kernel_time;
    *timing_arr = kernel_time;

    free(cImage);
    free(cOutImage);
    
    CUDA_CALL(cudaFree(gImage));
    CUDA_CALL(cudaFree(gOutImage));
    
    return output_checksum;
}


FLOATTYPE cudaMaxPooling(int c, int i_h, int i_w, int f_dim, double &sum, double &H2D_time, double &D2H_time, double *timing_arr)
{   
    double kernel_time = 0.0;

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
    
    // padding is always floor(F_dim/ 2) to ensure tensor output dims == tensor input dims
    int p_h = f_dim/ 2;
    int p_w = f_dim/ 2;

    int o_h = ((i_h + 2 * p_h - fh) / s_h) + 1;
    int o_w = ((i_w + 2 * p_w - fw) / s_w) + 1;

    #ifdef PRINT_PER_RUN
        printf("output dims: %d, %d\n", o_h, o_w);
    #endif
    
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
    #ifdef PRINT_PER_RUN
        printf("I = checksum: %f\n", input_checksum);
    #endif
    
    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    CUDA_CALL(cudaMemcpy(gImage, cImage, imageSize, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    H2D_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    #ifdef PRINT_PER_RUN
        printf("Copy host->dev %lf sec\n", H2D_time);
    #endif

    int shmem_size = sizeof(FLOATTYPE) * input_spatial_size;

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    kernel_max_pooling<<<c, input_spatial_size, shmem_size>>>(gImage, gOutImage, c, i_h, i_w, input_spatial_size, 
                                                            fw, fh, o_h, o_w, output_spatial_size);
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaGetLastError());
    kernel_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    #ifdef PRINT_PER_RUN
        printf("Time cuda code %lf sec\n", kernel_time);
    #endif

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    D2H_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    #ifdef PRINT_PER_RUN
        printf("Copy dev->host %lf sec\n", D2H_time);
    #endif

    FLOATTYPE output_checksum = calculateChecksum_float(cOutImage, c, o_h, o_w);
    #ifdef PRINT_PER_RUN
        printf("CUDA O = checksum: %f\n", output_checksum);
    #endif

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

    // update timing
    sum += kernel_time;
    *timing_arr = kernel_time;

    return output_checksum;
}


FLOATTYPE cudaMaxPooling_Ref(int c, int h, int w, int f_dim, double &sum, double &H2D_time, double &D2H_time, double *timing_arr)
{
    double kernel_time = 0.0;

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
    
    #ifdef PRINT_PER_RUN
        printf("I = checksum: %f\n", input_checksum);
    #endif

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    CUDA_CALL(cudaMemcpy(gImage, cImage, imageSize, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    H2D_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);

    #ifdef PRINT_PER_RUN
        printf("Copy host->dev %lf sec\n", H2D_time);
    #endif

    int shmem_size = sizeof(FLOATTYPE) * (TW + fw - 1) * (TH + fh - 1);
    dim3 blockDim(TW, TH);
    dim3 gridDim(DIV_RUP(w, TW), DIV_RUP(h, TH), c);

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    cudaMaxPool_ref<<<gridDim, blockDim, shmem_size>>>(gOutImage, gImage, c, h, w, fw, fh);
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaGetLastError());
    kernel_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    #ifdef PRINT_PER_RUN
        printf("Time cuda code %lf sec\n", kernel_time);
    #endif

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    D2H_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    #ifdef PRINT_PER_RUN
        printf("Copy dev->host %lf sec\n", D2H_time);
    #endif

    FLOATTYPE output_checksum = calculateChecksum_float(cOutImage, c, h, w);
    #ifdef PRINT_PER_RUN
        printf("CUDA O = checksum: %f\n", output_checksum);
    #endif

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

    // update timing
    sum += kernel_time;
    *timing_arr = kernel_time;

    return output_checksum;
}


FLOATTYPE cudnnMaxPooling(int c, int h, int w, int f_dim,
                    double &sum, double &H2D_time, double &D2H_time, double *timing_arr)
{
    double kernel_time = 0.0;
    int fw = f_dim;
    int fh = f_dim;

    long int imageSize = sizeof(FLOATTYPE) * c * w * h;

    FLOATTYPE *cImage = (FLOATTYPE *)malloc(imageSize);
    FLOATTYPE *gImage;

    // TODO: Change as per stride.
    long int outImageSize = sizeof(FLOATTYPE) * c * w * h; // DIV_RUP(w, fw) * DIV_RUP(h, fh);

    FLOATTYPE *cOutImage = (FLOATTYPE *)malloc(outImageSize);
    FLOATTYPE *gOutImage;

    struct timespec start, end;

    CUDA_CALL(cudaMalloc((void **)&gImage, imageSize));
    CUDA_CALL(cudaMalloc((void **)&gOutImage, outImageSize));

    CUDA_CALL(cudaMemset(gOutImage, 0, outImageSize));

    fillImage_floattype(cImage, c, h, w);

    #ifdef PRINT_PER_RUN
        printf("I = checksum: %lf\n", calculateChecksum_float(cImage, c, h, w));
    #endif

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    CUDA_CALL(cudaMemcpy(gImage, cImage, imageSize, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    H2D_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    
    #ifdef PRINT_PER_RUN
        printf("Copy host->dev %lf sec\n", H2D_time);
    #endif

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnPoolingDescriptor_t pooling_desc;
    // create descriptor handle
    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));

    // initialize descriptor
    checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            // descriptor handle
                                           CUDNN_POOLING_MAX,       // mode - max pooling
                                           CUDNN_NOT_PROPAGATE_NAN, // NaN propagation mode
                                           fh,                      // window height
                                           fw,                      // window width
                                           fh / 2,                  // vertical padding
                                           fw / 2,                  // horizontal padding
                                           1,                       // vertical stride
                                           1));                     // horizontal stride

    cudnnTensorDescriptor_t in_desc;
    // create input data tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    // initialize input data descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,           // descriptor handle
                                          CUDNN_TENSOR_NCHW, // data format
                                          CUDNN_DATA_FLOAT, // data type (precision)
                                          1,                 // number of images
                                          c,                 // number of channels
                                          h,                 // data height
                                          w));               // data width

    cudnnTensorDescriptor_t out_desc;
    // create output data tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
    // initialize output data descriptor
    checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,          // descriptor handle
                                          CUDNN_TENSOR_NCHW, // data format
                                          CUDNN_DATA_FLOAT, // data type (precision)
                                          1,                 // number of images
                                          c,                 // number of channels
                                          h,                 // data height
                                          w));               // data width

    // Scaling factor
    FLOATTYPE alpha = 1.0;
    FLOATTYPE beta = 0.0;

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    // Call pooling operator
    checkCUDNN(cudnnPoolingForward(cudnn,        // cuDNN context handle
                                   pooling_desc, // pooling descriptor handle
                                   &alpha,       // alpha scaling factor
                                   in_desc,      // input tensor descriptor
                                   gImage,       // input data pointer to GPU memory
                                   &beta,        // beta scaling factor
                                   out_desc,     // output tensor descriptor
                                   gOutImage));  // output data pointer from GPU memory
    cudaDeviceSynchronize();
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    kernel_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    #ifdef PRINT_PER_RUN
        printf("Time cudnn code %lf sec\n", kernel_time);
    #endif

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    
    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }

    D2H_time = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    #ifdef PRINT_PER_RUN
        printf("Copy dev->host %lf sec\n", D2H_time);
    #endif

    FLOATTYPE output_checksum = calculateChecksum_float(cOutImage, c, h, w);

    #ifdef PRINT_PER_RUN
        printf("CUDNN O = checksum: %lf\n", output_checksum);
    #endif

    free(cImage);
    free(cOutImage);
    CUDA_CALL(cudaFree(gImage));
    CUDA_CALL(cudaFree(gOutImage));

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudnnDestroy(cudnn);

    // update timing
    sum += kernel_time;
    *timing_arr = kernel_time;

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
    
    int CASE_SELECT = atoi(argv[6]);

    double H2D_ref = 0.0;
    double H2D_kernel = 0.0;

    double D2H_ref = 0.0;
    double D2H_kernel = 0.0;

    double sum_ref = 0.0;
    double sum1_ref = 0.0;
    double *timing_arr_ref;

    double sum_kernel = 0.0;
    double sum1_kernel = 0.0;
    double *timing_arr_kernel;

    timing_arr_ref = (double *)calloc(NUM_RUNS, sizeof(double));
    timing_arr_kernel = (double *)calloc(NUM_RUNS, sizeof(double));

    double average_ref, variance_ref, std_dev_ref;
    double average_kernel, variance_kernel, std_dev_kernel;

    // profile normal case 3 kernel
    if(CASE_SELECT == 0){
        printf("C:%d; H:%d; W:%d; F_dim:%d; padding:%d\n", C, H, W, F_dim, F_dim/ 2);
        for(int run=0; run < NUM_RUNS; run++){    
            // printf("Reference Max Pool Using cuDNN\n");
            // internally handles padding logic
            FLOATTYPE ref_checksum = cudnnMaxPooling(C, H, W, F_dim, sum_ref, H2D_ref, D2H_ref, timing_arr_ref + run);
            // FLOATTYPE ref_checksum = cudaMaxPooling_Ref(C, H, W, F_dim, sum_ref, H2D_ref, D2H_ref, timing_arr_ref + run);
            
            // printf("FC2: Max Pool Using CUDA\n");
            FLOATTYPE kernel_checksum = cudaMaxPooling(C, H, W, F_dim, sum_kernel, H2D_kernel, D2H_kernel, timing_arr_kernel + run);

            assert(ref_checksum == kernel_checksum);
            // printf("==============================\n\n");
        }

        /*  Compute average */
        average_ref = sum_ref / (double) NUM_RUNS;
        average_kernel = sum_kernel / (double) NUM_RUNS;

        /*  Compute  variance  and standard deviation  */
        for (unsigned long long i = 0; i < NUM_RUNS; i++)
        {
            sum1_ref = sum1_ref + pow((*(timing_arr_ref + i) - average_ref), 2);
            sum1_kernel = sum1_kernel + pow((*(timing_arr_kernel + i) - average_kernel), 2);
        }

        variance_ref = sum1_ref / (double)NUM_RUNS;
        variance_kernel = sum1_kernel / (double)NUM_RUNS;
        std_dev_ref = sqrt(variance_ref);
        std_dev_kernel = sqrt(variance_kernel);
        
        printf("Runs: %llu\n\r", NUM_RUNS);
        printf("Reference Timings:\n");
        printf("H2D copy (ms): %lf\n\r", H2D_ref);
        printf("Kernel Average (ms): %lf\n\r", average_ref);
        printf("Kernel Variance (ms): %lf\n\r", variance_ref);
        printf("Kernel Std Dev (ms): %lf\n\r", std_dev_ref);
        printf("D2H copy (ms): %lf\n\r", D2H_ref);

        printf("Kernel Compute Timings:\n");
        printf("H2D copy (ms): %lf\n\r", H2D_kernel);
        printf("Average (ms): %lf\n\r", average_kernel);
        printf("Variance (ms): %lf\n\r", variance_kernel);
        printf("Std Dev (ms): %lf\n\r", std_dev_kernel);
        printf("D2H copy (ms): %lf\n\r", D2H_kernel);
        printf("==================================================================\n");

    }
    else if(CASE_SELECT == 1){
        printf("Profile Integrated case\n");
        FLOATTYPE ref_checksum_arr[3] = {0.0, 0.0, 0.0};

        int F_dim_arr[3] = {5, 9, 13};

        // profile integrated case
        for(int run=0; run < NUM_RUNS; run++){
            // printf("Reference Max Pool Using CUDA\n");
            D2H_ref = 0.0;  // reset across runs; use one run as typ.
            H2D_ref = 0.0;  // reset across runs; use one run as typ.

            for(int dim_iter=0; dim_iter< 3; dim_iter++){
                double D2H_ref_local = 0.0;
                double H2D_ref_local = 0.0;
                ref_checksum_arr[dim_iter] = cudnnMaxPooling(C, H, W, F_dim_arr[dim_iter], sum_ref, H2D_ref_local, D2H_ref_local, timing_arr_ref + run);
                D2H_ref += D2H_ref_local;
                H2D_ref += H2D_ref_local;
            }

            // printf("==============================\n\n");
            // printf("Integrated Max Pool\n");
            FLOATTYPE kernel_checksum = cudaMaxPoolingIntegrated(C, H, W, F_dim_arr, sum_kernel, H2D_kernel, D2H_kernel, timing_arr_kernel + run);
            
            FLOATTYPE ref_checksum_cumm = 0.0;
            for(int dim_iter=0; dim_iter< 3; dim_iter++){
                ref_checksum_cumm += ref_checksum_arr[dim_iter];
            }
            assert(ref_checksum_cumm == kernel_checksum);
        }

        /*  Compute average */
        average_ref = sum_ref / (double) NUM_RUNS;
        average_kernel = sum_kernel / (double) NUM_RUNS;

        /*  Compute  variance  and standard deviation  */
        for (unsigned long long i = 0; i < NUM_RUNS; i++)
        {
            sum1_ref = sum1_ref + pow((*(timing_arr_ref + i) - average_ref), 2);
            sum1_kernel = sum1_kernel + pow((*(timing_arr_kernel + i) - average_kernel), 2);
        }

        variance_ref = sum1_ref / (double)NUM_RUNS;
        variance_kernel = sum1_kernel / (double)NUM_RUNS;
        std_dev_ref = sqrt(variance_ref);
        std_dev_kernel = sqrt(variance_kernel);
        
        printf("Runs: %llu\n\r", NUM_RUNS);
        printf("Reference Timings:\n");
        printf("H2D copy (ms): %lf\n\r", H2D_ref);
        printf("Kernel Average (ms): %lf\n\r", average_ref);
        printf("Kernel Variance (ms): %lf\n\r", variance_ref);
        printf("Kernel Std Dev (ms): %lf\n\r", std_dev_ref);
        printf("D2H copy (ms): %lf\n\r", D2H_ref);

        printf("Kernel Compute Timings:\n");
        printf("H2D copy (ms): %lf\n\r", H2D_kernel);
        printf("Kernel Average (ms): %lf\n\r", average_kernel);
        printf("Kernel Variance (ms): %lf\n\r", variance_kernel);
        printf("Kernel Std Dev (ms): %lf\n\r", std_dev_kernel);
        printf("D2H copy (ms): %lf\n\r", D2H_kernel);
        printf("==================================================================\n");
    }

    // free timing DS
    free(timing_arr_ref);
    free(timing_arr_kernel);
    return 0;
}
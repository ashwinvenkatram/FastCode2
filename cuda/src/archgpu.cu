#include <iostream>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "../include/utils.h"

using namespace std;


__global__ void integrated_kernel_max_pooling(FLOATTYPE *in, FLOATTYPE *out0, FLOATTYPE *out1, FLOATTYPE *out2, int c, 
                                    int i_h, int i_w, int input_spatial_size,
                                    int f_w0, int f_h0, int f_w1, int f_h1, int f_w2, int f_h2, int o_h, int o_w, int output_spatial_size)
{
    // 0: f_dim 5
    // 1: f_dim 9
    // 2: f_dim 13

    // Determine the thread and block index
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    // Outer loop un-necessary; tidx implicitly account for the centre of the receptive field
    
    // Calculate the spatial row and column index
    // invariant to channel iteration
    int row = tidx / i_w;
    int col = tidx % i_h;

    // if thread is accessing out-of-bounds, return
    if (row < 0 || row >= i_h || col < 0 || col >= i_w)
    {
        return;
    }

    // Iterate over the input channel dim
    for (int c_iter = 0; c_iter < c; c_iter++){

        // Initialize the max value to the minimum float value
        FLOATTYPE max_val0 = 0.0;
        FLOATTYPE max_val1 = 0.0;
        FLOATTYPE max_val2 = 0.0;

        int input_channel_offset = c_iter * input_spatial_size;
        int output_channel_offset = c_iter * output_spatial_size;

        // Iterate over the pooling window
        for (int i = -f_w2 / 2; i <= f_w2 / 2; i++) {
            for (int j = -f_h2 / 2; j <= f_h2 / 2; j++) {
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
}


__global__ void kernel_max_pooling(FLOATTYPE *in, FLOATTYPE *out, int c, 
                                    int i_h, int i_w, int input_spatial_size,
                                    int f_w, int f_h, int o_h, int o_w, int output_spatial_size)
{
    // Determine the thread and block index
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    // Outer loop un-necessary; tidx implicitly account for the centre of the receptive field
    
    // Calculate the spatial row and column index
    // invariant to channel iteration
    int row = tidx / i_w;
    int col = tidx % i_h;

    // if thread is accessing out-of-bounds, return
    if (row < 0 || row >= i_h || col < 0 || col >= i_w)
    {
        return;
    }

    // Iterate over the input channel dim
    for (int c_iter = 0; c_iter < c; c_iter++){

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


void cudaMaxPoolingIntegrated(int c, int i_h, int i_w, int* f_dim, FLOATTYPE *out_checksum_arr)
{   
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

    printf("output0 dims: %d, %d\n", o_h0, o_w0);
    printf("output1 dims: %d, %d\n", o_h1, o_w1);
    printf("output2 dims: %d, %d\n", o_h2, o_w2);
    
    // ensure that all output spatial dimensions are equal
    // Requirement for kernel design
    assert(o_h0==o_h1 && o_h1==o_h2);
    assert(o_w0==o_w1 && o_w1==o_w2);

    int output_spatial_size0 = o_w0 * o_h0;
    long int outImageSize0 = sizeof(FLOATTYPE) * c * output_spatial_size0;
    
    FLOATTYPE *cOutImage0 = (FLOATTYPE *)malloc(outImageSize0);
    FLOATTYPE *cOutImage1 = (FLOATTYPE *)malloc(outImageSize0);
    FLOATTYPE *cOutImage2 = (FLOATTYPE *)malloc(outImageSize0);
    
    FLOATTYPE *gOutImage0;
    FLOATTYPE *gOutImage1;
    FLOATTYPE *gOutImage2;
    
    struct timespec start, end;

    CUDA_CALL(cudaMalloc((void **)&gImage, imageSize));
    CUDA_CALL(cudaMalloc((void **)&gOutImage0, outImageSize0));
    CUDA_CALL(cudaMalloc((void **)&gOutImage1, outImageSize0));
    CUDA_CALL(cudaMalloc((void **)&gOutImage2, outImageSize0));

    CUDA_CALL(cudaMemset((void *)gOutImage0, 0, outImageSize0));
    CUDA_CALL(cudaMemset((void *)gOutImage1, 0, outImageSize0));
    CUDA_CALL(cudaMemset((void *)gOutImage2, 0, outImageSize0));

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

    // int shmem_size = sizeof(FLOATTYPE) * (TW + fw - 1) * (TH + fh - 1);
    // dim3 blockDim(TW, TH);
    // dim3 gridDim(DIV_RUP(i_w, TW), DIV_RUP(i_h, TH), c);
    dim3 gridDim(1);
    dim3 blockDim(192);

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    // integrated_kernel_max_pooling<<<gridDim, blockDim>>>(gImage, gOutImage0, gOutImage1, gOutImage2, c, i_h, i_w, input_spatial_size, 
    //                                                         fw0, fh0, fw1, fh1, fw2, fh2, o_h0, o_w0, output_spatial_size0);
    kernel_max_pooling<<<gridDim, blockDim>>>(gImage, gOutImage0, c, i_h, i_w, input_spatial_size, 
                                                fw0, fh0, o_h0, o_w0, output_spatial_size0);
    
    kernel_max_pooling<<<gridDim, blockDim>>>(gImage, gOutImage1, c, i_h, i_w, input_spatial_size, 
                                                fw1, fh1, o_h0, o_w0, output_spatial_size0);

    kernel_max_pooling<<<gridDim, blockDim>>>(gImage, gOutImage2, c, i_h, i_w, input_spatial_size, 
                                                fw2, fh2, o_h0, o_w0, output_spatial_size0);                                                
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
    CUDA_CALL(cudaMemcpy(cOutImage0, gOutImage0, outImageSize0, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cOutImage1, gOutImage1, outImageSize0, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cOutImage2, gOutImage2, outImageSize0, cudaMemcpyDeviceToHost));

    if (clock_gettime(CLOCK_MONOTONIC, &end))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    printf("Copy dev->host %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

    FLOATTYPE output_checksum0 = calculateChecksum_float(cOutImage0, c, o_h0, o_w0);
    FLOATTYPE output_checksum1 = calculateChecksum_float(cOutImage1, c, o_h0, o_w0);
    FLOATTYPE output_checksum2 = calculateChecksum_float(cOutImage2, c, o_h0, o_w0);

    printf("CUDA O = checksum: %f\n", output_checksum0);
    printf("CUDA O = checksum: %f\n", output_checksum1);
    printf("CUDA O = checksum: %f\n", output_checksum2);
    
    out_checksum_arr[0] = output_checksum0;
    out_checksum_arr[1] = output_checksum1;
    out_checksum_arr[2] = output_checksum2;

    #ifdef PRINT_DEBUG
        printf("Input tensor:\n");
        print_tensor(cImage, c, i_h, i_w);
        printf("Output tensor:\n");
        print_tensor(cOutImage, c, o_h, o_w);
    #endif

    free(cImage);
    free(cOutImage0);
    free(cOutImage1);
    free(cOutImage2);
    CUDA_CALL(cudaFree(gImage));
    CUDA_CALL(cudaFree(gOutImage0));
    CUDA_CALL(cudaFree(gOutImage1));
    CUDA_CALL(cudaFree(gOutImage2));
}


FLOATTYPE cudaMaxPooling(int c, int i_h, int i_w, int f_dim)
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
    
    // padding is always floor(F_dim/ 2) to ensure tensor output dims == tensor input dims
    int p_h = f_dim/ 2;
    int p_w = f_dim/ 2;

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

    // int shmem_size = sizeof(FLOATTYPE) * (TW + fw - 1) * (TH + fh - 1);
    // dim3 blockDim(TW, TH);
    // dim3 gridDim(DIV_RUP(i_w, TW), DIV_RUP(i_h, TH), c);
    dim3 gridDim(1);
    dim3 blockDim(192);

    if (clock_gettime(CLOCK_MONOTONIC, &start))
    {
        printf("CLOCK ERROR. Exiting.\n");
        std::exit(EXIT_FAILURE);
    }
    
    kernel_max_pooling<<<gridDim, blockDim>>>(gImage, gOutImage, c, i_h, i_w, input_spatial_size, 
                                                            fw, fh, o_h, o_w, output_spatial_size);

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
    
    int CASE_SELECT = atoi(argv[6]);

    // profile normal case 3 kernel
    if(CASE_SELECT == 0){
        for(int run=0; run < NUM_RUNS; run++){
            printf("C:%d; H:%d; W:%d; F_dim:%d; padding:%d\n", C, H, W, F_dim, F_dim/ 2);
            printf("Reference Max Pool Using CUDA\n");
            // internally handles padding logic
            FLOATTYPE ref_checksum = cudaMaxPooling_Ref(C, H, W, F_dim);
            printf("\n");

            printf("FC2: Max Pool Using CUDA\n");
            // requires padding as input & perform correction
            // (int padding, int c, int i_h, int i_w, int fw, int fh)
            FLOATTYPE kernel_checksum = cudaMaxPooling(C, H, W, F_dim);

            assert(ref_checksum == kernel_checksum);
            printf("==============================\n\n");
        }
    }
    else if(CASE_SELECT == 1){
        FLOATTYPE ref_checksum_arr[3] = {0.0, 0.0, 0.0};
        FLOATTYPE kernel_checksum_arr[3] = {0.0, 0.0, 0.0};

        int F_dim_arr[3] = {5, 9, 13};

        // profile integrated case
        for(int run=0; run < NUM_RUNS; run++){
            printf("Reference Max Pool Using CUDA\n");
            for(int dim_iter=0; dim_iter< 3; dim_iter++){
                ref_checksum_arr[dim_iter] = cudaMaxPooling_Ref(C, H, W, F_dim_arr[dim_iter]);
            }

            printf("==============================\n\n");
            printf("Integrated Max Pool\n");
            cudaMaxPoolingIntegrated(C, H, W, F_dim_arr, kernel_checksum_arr);
            
            for(int dim_iter=0; dim_iter< 3; dim_iter++){
                if(ref_checksum_arr[dim_iter] == kernel_checksum_arr[dim_iter]){
                    printf("Kernel: %d\t PASS\n", F_dim_arr[dim_iter]);
                } else{
                    printf("Kernel: %d\t FAIL\n", F_dim_arr[dim_iter]);
                }
            }

        }
    }

    return 0;
}
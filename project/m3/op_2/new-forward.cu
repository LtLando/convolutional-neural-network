#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 32

__global__ void unroll(float *input_unroll, const float *input, const int b, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int H_unroll = Channel*K*K;
    const int W_unroll = Height_out*Width_out;
    
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < Channel * W_unroll) {
        int c = idx / W_unroll;
        int s = idx % W_unroll;
        int h_out = s / Width_out;
        int w_out = s % Width_out;
        for (int p = 0; p < K; p++)
            for (int q = 0; q < K; q++) {
                int h_unroll = c * K * K + p * K + q;
                int w_unroll = h_out * Width_out + w_out;
                input_unroll[h_unroll*W_unroll + w_unroll] = in_4d(b, c, h_out + p, w_out + q);
            }
    }

    #undef in_4d
}

__global__ void conv_forward_kernel(float *output, const float *input_unroll, const float *mask, const int b, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int H_unroll = Channel*K*K;
    const int W_unroll = Height_out*Width_out;
    const int H_mask = Map_out;
    const int W_mask = H_unroll;

    __shared__ float mask_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float input_tile[TILE_WIDTH][TILE_WIDTH];
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    float p = 0;

    for (int i = 0; i < ceil((float) W_mask/TILE_WIDTH); i++) {
        if ((row < H_mask) && (i*TILE_WIDTH+threadIdx.x < W_mask)) {
            mask_tile[threadIdx.y][threadIdx.x] = mask[row*W_mask + i*TILE_WIDTH+threadIdx.x];
        }
        else mask_tile[threadIdx.y][threadIdx.x] = 0.0;

        if ((col < W_unroll) && (i*TILE_WIDTH+threadIdx.y < H_unroll)) {
            input_tile[threadIdx.y][threadIdx.x] = input_unroll[(i*TILE_WIDTH+threadIdx.y)*W_unroll + col];
        }
        else input_tile[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();
        for (int j = 0; j < TILE_WIDTH; j++) p += mask_tile[threadIdx.y][j] * input_tile[j][threadIdx.x];
        __syncthreads();
    }

    if (row < H_mask && col < W_unroll) output[row*W_unroll + col] = p;
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int H_unroll = Channel*K*K;
    const int W_unroll = Height_out*Width_out;
    const int H_mask = Map_out;
    float *input_unroll;

    cudaMalloc((void **) &input_unroll, H_unroll * W_unroll * sizeof(float));

    dim3 DimGridUnroll(ceil((float) Channel*Height_out*Width_out / BLOCK_SIZE), 1, 1);
    dim3 DimBlockUnroll(BLOCK_SIZE, 1, 1);
    dim3 DimGridMult(ceil((float) W_unroll/TILE_WIDTH), ceil((float) H_mask/TILE_WIDTH), 1);
    dim3 DimBlockMult(TILE_WIDTH, TILE_WIDTH, 1);

    for (int b = 0; b < Batch; b++) {
        int out_off = b * Map_out * Height_out * Width_out;

        unroll<<<DimGridUnroll, DimBlockUnroll>>>(input_unroll, device_input, b, Channel, Height, Width, K);
        cudaDeviceSynchronize();
        conv_forward_kernel<<<DimGridMult, DimBlockMult>>>(device_output + out_off, input_unroll, device_mask, b, Map_out, Channel, Height, Width, K);
        cudaDeviceSynchronize();
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
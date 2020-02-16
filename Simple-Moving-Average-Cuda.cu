
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

#define THREAD_PER_BLOCK 16

void simple_moving_Average(const int array_size, const float* datapoints_array, const int N, float* output_array)
{
    int i, j;
    float sum;
    for (i = 0; i < N - 1; i++) {
        output_array[i] = datapoints_array[i];
        //printf("Simple Moving Average : CPU : %f \n", output_array[i]);
    }
    for (; i < array_size; i++) {
        sum = 0;
        for (j = 0; j < N; j++) {
            sum += datapoints_array[i - j];
        }
        output_array[i] = sum / N;
        //printf("Simple Moving Average : CPU : %f \n", output_array[i]);
    }
}

__global__ void simple_moving_Average_kernel(const float array_size, const float* datapoints_array, const int N, float* output_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, j;
    float sum;
    if (i < N - 1) {
        output_array[i] = datapoints_array[i];
        //printf("Simple Moving Average : GPU: %f \n", output_array[i]);
    }
    else if (i < array_size) {
        sum = 0;
        for (j = 0; j < N; j++) {
            sum += datapoints_array[i - j];
        }
        output_array[i] = sum / N;
        //printf("Simple Moving Average : GPU: %f \n", output_array[i]);
    }
}

void simple_moving_Average_gpu(const int array_size, const float* datapoints_array, const int N, float* output_array)
{
    float* datapoints_array_gpu, * output_array_gpu;
    cudaMalloc((void**)&datapoints_array_gpu, array_size * sizeof(float));
    cudaMalloc((void**)&output_array_gpu, array_size * sizeof(float));
    cudaMemcpy(datapoints_array_gpu, datapoints_array, array_size * sizeof(float), cudaMemcpyHostToDevice);

    //Simple Moving Average GPU
    clock_t start_gpu, end_gpu;
    float total_time;
    start_gpu = clock();
    dim3 blocks(THREAD_PER_BLOCK);
    dim3 grids(array_size / THREAD_PER_BLOCK + 1);
    simple_moving_Average_kernel << <grids, blocks >> > (array_size, datapoints_array_gpu, N, output_array_gpu);
    cudaMemcpy((void*)output_array, (void*)output_array_gpu, array_size * sizeof(float), cudaMemcpyDeviceToHost);
    end_gpu = clock();
    //time count stops 
    total_time = ((float)(end_gpu - start_gpu)) / CLOCKS_PER_SEC;
    //calulate total time
    printf("\nTime taken to calculate moving average for GPU: %f \n", total_time);

    cudaFree(datapoints_array_gpu);
    cudaFree(output_array_gpu);
}

int main()
{
    const int input_array_size = 10000;
    float* sample_array = new float[input_array_size];
    float* output = new float[input_array_size];
    float total_time;
    int N;

    printf("Enter N value:\n");
    scanf("%d", &N);

    //Simple Moving Average CPU
    clock_t start, end;
    start = clock();
    
    for (int i = 0; i < input_array_size; ++i) {
        sample_array[i] = rand() % 100;
    }

    simple_moving_Average(input_array_size, sample_array, N, output);
    end = clock();
    //time count stops 
    total_time = ((float)(end - start)) / CLOCKS_PER_SEC;
    //calulate total time
    printf("\nTime taken to calculate moving average for CPU: %f\n", total_time);

    //Simple Moving Average GPU
    simple_moving_Average_gpu(input_array_size, sample_array, N, output);
   

    return 0;
}


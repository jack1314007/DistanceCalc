/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "DistanceCalc.h"
#include "GetDeviceInfo.h"
#include "math.h"

#define TILE_WIDTH 16
#define INF 1.0/0.0;

/*
 * Input:
 */

__device__ float CalcTuple1(float n1, float n2){
	float out = 0;
	out = (n1-n2)*(n1-n2);
	return out;
	}

__device__ float MinOf3(float n1, float n2, float n3)
{
	float min = INF;
	if (n1 < min)
		min = n1;
	if (n2 < min)
		min = n2;
	if (n3 < min)
		min = n3;
	return min;
}

__device__ float MaxOf2(float n1, float n2){
	float max = -1;
	if (n1>max)
		max = n1;
	if (n2>max)
		max = n2;
	return max;
}

bool isValueInArray(float val, float *arr, int size){
    for (int i=0; i < size; i++) {
        if (arr[i] == val)
            return true;
    }
    return false;
}



__global__ void euclidian_distance(float* in1, float* in2, float* out, int row, int col)
{
	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

	float outValue = 0;
	if(i<row&&j<row&&i<=j){
	for(int k = 0;k<col;k++)
	{
		outValue += CalcTuple1(in1[i*col+k], in2[j*col+k]);
	}
	outValue = sqrt(outValue);
	//printf("Matrix[%d] and Matrix[%d]: %f \n",i,j,outValue);
	out[i*row + j] = outValue;
	out[j*row + i] = outValue;
	}
}


__global__ void manhattan_distance(float* in1, float* in2, float* out, int row, int col)
{

	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;
	float outValue = 0;
	if(i<row&&j<row&&i<=j){
	for(int k = 0;k<col;k++)
	{
		outValue += abs(in1[i*col+k] - in2[j*col+k]);

	}
	//printf("Matrix[%d] and Matrix[%d]: %f \n",i,j,outValue);
	out[i*row + j] = outValue;
	out[j*row + i] = outValue;
	}
}

__global__ void DTW_distance(float* in1, float* in2, float* out, int row1, int row2, int col)
{
	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;
	float outValue = -1;
	float dist = 0;
	if(i<row1 && j<row2 && out[i*row2+j] == -1)
	{
		if(i == 0 && j == 0)
		{
			outValue = 0;
		}
		else if(i == 0 && j != 0)
		{
			outValue = INF;
		}
		else if(j == 0 && i != 0)
		{
			outValue = INF;
		}
		else if ((out[(i-1)*row2 + j] != -1)
				&& (out[i * row2 + j-1] != -1)
				&&(out[(i-1) * row2 + j-1] != -1))
		{
			for(int k = 0;k<col;k++)
				{
					dist += CalcTuple1(in1[i*col+k], in2[j*col+k]);
				}
			dist = sqrt(dist);
			outValue = dist + MinOf3(out[(i-1)*row2 + j],out[i * row2 + j-1],out[(i-1) * row2 + j-1]);
		}
		out[i * row2 + j] = outValue;
	}
}

__global__ void frechet_distance(float* in1, float* in2, float* out, int row1, int row2, int col)
{
	int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int j = blockIdx.x * TILE_WIDTH + threadIdx.x;
	float outValue = -1;
	float dist = 0;
	if(i<row1 && j<row2 && out[i*row2+j] == -1)
	{
		for(int k = 0;k<col;k++)
			{
				dist += CalcTuple1(in1[i*col+k], in2[j*col+k]);
			}
		dist = sqrt(dist);

		if(i == 0 && j == 0)
		{
			outValue = dist;
		}
		else if(i == 0 && j != 0 && (out[i*row2 + j-1] != -1))
		{
			outValue = MaxOf2(dist,out[i*row2 + j-1]);
		}
		else if(j == 0 && i != 0 && (out[(i-1)*row2 + j] != -1))
		{
			outValue = MaxOf2(dist,out[(i-1)*row2 + j]);
		}
		else if ((out[(i-1)*row2 + j] != -1)
				&& (out[i * row2 + j-1] != -1)
				&&(out[(i-1) * row2 + j-1] != -1))
		{

			outValue = MaxOf2(dist,MinOf3(out[(i-1)*row2 + j],out[i * row2 + j-1],out[(i-1) * row2 + j-1]));
		}
		out[i * row2 + j] = outValue;
	}
}



void core_calculation(float *in1,float *in2,float *out, int row1, int row2, int col, int algorithm, int device_id)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;
	int row = 0;
	err = cudaSetDevice(device_id);
	if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to find the device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	getGpuInfo();

	// size of input matrix A and B
	size_t size_input1 = row1 * col * sizeof(float);
	size_t size_input2 = row2 * col * sizeof(float);

	// size of output matrix C
	size_t size_output = row1 * row2 * sizeof(float);

	// Allocate the device input vector A
	float *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size_input1);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size_input2);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float *d_C = NULL;
	err = cudaMalloc((void **)&d_C, size_output);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in device memory
	printf("\nCopy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, in1, size_input1, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_B, in2, size_input2, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_C, out, size_output, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	//Define the dimension of the Grid and the Block
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 blocksPerGrid(row/TILE_WIDTH + 1,row/TILE_WIDTH + 1);

	switch(algorithm) {
	    case 1: printf("Calculating Euclidian Distance\n"); // initialization
	    		printf("Calculating...\n");
	    		if(row1 != row2)
	    			{
	    				printf("The length of these 2 matrix is not same\n");
	    				exit(0);
	    			}
	    			else
	    				row = row1;
	    		euclidian_distance<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,row,col);
	    		cudaDeviceSynchronize();
	    		// Copy the output vector C in device memory to the output vectors in host memory
	    		printf("Copy input data from the device memory to the host device\n");
	    		err = cudaMemcpy(out, d_C, size_output, cudaMemcpyDeviceToHost);
	    		if (err != cudaSuccess)
	    			{
	    				fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
	    				exit(EXIT_FAILURE);
	    			}
	    		cudaFree(d_A);
	    		cudaFree(d_B);
	    		cudaFree(d_C);
	            break;
	    case 2: printf("Calculating Manhattan Distance\n"); // initialization
	    		printf("Calculating...\n");
	    		if(row1 != row2)
	    		{
	    			printf("The length of these 2 matrix is not same\n");
	    			exit(0);
	    		}
	    		else
	    			row = row1;
	    		manhattan_distance<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,row,col);
	    		cudaDeviceSynchronize();
	    		// Copy the output vector C in device memory to the output vectors in host memory
	    		printf("Copy input data from the device memory to the host device\n");
	    		err = cudaMemcpy(out, d_C, size_output, cudaMemcpyDeviceToHost);
	    		if (err != cudaSuccess)
	    			{
	    				fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
	    				exit(EXIT_FAILURE);
	    			}
	    		cudaFree(d_A);
	    		cudaFree(d_B);
	    		cudaFree(d_C);
	            break;

	    case 3: printf("Calculating DTW Distance\n");
	    		printf("Calculating...\n");
	    		while(isValueInArray(-1,out,row1*row2)){
		    		DTW_distance<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,row1,row2,col);
		    		cudaDeviceSynchronize();
		    		err = cudaMemcpy(out, d_C, size_output, cudaMemcpyDeviceToHost);
		    		if (err != cudaSuccess)
		    			{
		    				fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		    				exit(EXIT_FAILURE);
		    			}
	    		}
	    		// Copy the output vector C in device memory to the output vectors in host memory
	    		printf("Copy input data from the device memory to the host device\n");
	    		cudaFree(d_A);
	    		cudaFree(d_B);
	    		cudaFree(d_C);
	            break;

	    case 4: printf("Calculating Frechet Distance\n");
	    		printf("Calculating...\n");
	    		while(isValueInArray(-1,out,row1*row2)){
	    			frechet_distance<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,row1,row2,col);
		    		cudaDeviceSynchronize();
		    		err = cudaMemcpy(out, d_C, size_output, cudaMemcpyDeviceToHost);
		    		if (err != cudaSuccess)
		    			{
		    				fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		    				exit(EXIT_FAILURE);
		    			}
	    		}
	    		// Copy the output vector C in device memory to the output vectors in host memory
	    		printf("Copy input data from the device memory to the host device\n");
	    		cudaFree(d_A);
	    		cudaFree(d_B);
	    		cudaFree(d_C);
	            break;
	    default:printf("This Method Has Not Been Implemented Yet\n");
	    		printf("Reference Values:\n");
	    		printf("\t Euclidian Distance: 1\n");
	    		printf("\t Manhattan Distance: 1\n");
	            break;
	}


}



void distance_calculation(float *in1,float *in2,float *out, int row1, int row2, int col, int algorithm)
{

	core_calculation( in1, in2, out, row1, row2, col, algorithm, 0);
}


/*
 * getDeviceInfo.cpp
 *
 *  Created on: Jun 4, 2015
 *      Author: jackzhang
 */

#include "getDeviceInfo.h"

t_device_info *deviceInfo;

template <class T>
  void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute, int device)
{
    CUresult error =    cuDeviceGetAttribute(attribute, device_attribute, device);

    if (CUDA_SUCCESS != error)
    {
        fprintf(stderr, "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
                error, __FILE__, __LINE__);

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void printDevInfo(t_device_info deviceInfo)
{
	printf("Device ID: %d\n", deviceInfo.device_id);
	printf("Total amount of global memory: %llu (bytes)\n",deviceInfo.total_global_memory);
	printf("Total number of CUDA Cores: %llu \n",deviceInfo.multi_processor_count);
	printf("Total amount of constant memory: %lu (bytes)\n",deviceInfo.total_constant_memory);
	printf("Total amount of shared memory per block: %lu (bytes)\n",deviceInfo.shared_memory_per_block);
	printf("Maximum number of threads per multiprocessor: %du (bytes)\n",deviceInfo.max_thread_number_per_multiprocessor);
	printf("Maximum number of threads per block: %du (bytes)\n",deviceInfo.max_thread_number_per_block);
	printf("Maximum Texture Dimension Size (x,y,z): 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
			deviceInfo.max_texture_dimension_1D,deviceInfo.max_texture_dimension_2D[0],deviceInfo.max_texture_dimension_2D[1],
			deviceInfo.max_texture_dimension_3D[0],deviceInfo.max_texture_dimension_3D[1],deviceInfo.max_texture_dimension_3D[2]);
	printf("Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceInfo.dimension_size_of_thread[0],deviceInfo.dimension_size_of_thread[1],deviceInfo.dimension_size_of_thread[2]);
	printf("Max dimension size of a grid size (x,y,z): (%d, %d, %d)\n",
			deviceInfo.dimension_size_of_grid[0],deviceInfo.dimension_size_of_grid[1],deviceInfo.dimension_size_of_grid[2]);
}

void getGpuInfo()
{
	printf("Retrieving the GPU information of this machine.\n");
	int deviceCount = 0;

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess)
	{
	    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
	    printf("Result = FAIL\n");
	    exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
	    printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
	    printf("Detected %d CUDA Capable device(s)\n\n", deviceCount);
	}

	deviceInfo = new t_device_info[deviceCount];

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		deviceInfo[dev].device_id = dev;
		deviceInfo[dev].total_global_memory = deviceProp.totalGlobalMem;
		deviceInfo[dev].multi_processor_count = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *deviceProp.multiProcessorCount;
		deviceInfo[dev].total_constant_memory = deviceProp.totalConstMem;
		deviceInfo[dev].shared_memory_per_block = deviceProp.sharedMemPerBlock;
		deviceInfo[dev].max_thread_number_per_block = deviceProp.maxThreadsPerBlock;
		deviceInfo[dev].max_thread_number_per_multiprocessor = deviceProp.maxThreadsPerMultiProcessor;
		deviceInfo[dev].max_texture_dimension_1D = deviceProp.maxTexture1D;
		deviceInfo[dev].max_texture_dimension_2D[0] = deviceProp.maxTexture2D[0];
		deviceInfo[dev].max_texture_dimension_2D[1] = deviceProp.maxTexture2D[1];
		deviceInfo[dev].max_texture_dimension_3D[0] = deviceProp.maxTexture3D[0];
		deviceInfo[dev].max_texture_dimension_3D[1] = deviceProp.maxTexture3D[1];
		deviceInfo[dev].max_texture_dimension_3D[2] = deviceProp.maxTexture3D[2];
		deviceInfo[dev].dimension_size_of_thread[0] = deviceProp.maxThreadsDim[0];
		deviceInfo[dev].dimension_size_of_thread[1] = deviceProp.maxThreadsDim[1];
		deviceInfo[dev].dimension_size_of_thread[2] = deviceProp.maxThreadsDim[2];
		deviceInfo[dev].dimension_size_of_grid[0] = deviceProp.maxGridSize[0];
		deviceInfo[dev].dimension_size_of_grid[1] = deviceProp.maxGridSize[1];
		deviceInfo[dev].dimension_size_of_grid[2] = deviceProp.maxGridSize[2];
		printDevInfo(deviceInfo[dev]);
	}

}




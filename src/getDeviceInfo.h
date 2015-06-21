/*
 * getDeviceInfo.h
 *
 *  Created on: Jun 4, 2015
 *      Author: jackzhang
 */

#ifndef GETDEVICEINFO_H_
#define GETDEVICEINFO_H_

#include <memory>
#include <iostream>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>


inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

typedef struct{
	unsigned int device_id;
	unsigned long long total_global_memory;
	unsigned long long multi_processor_count;
	unsigned long total_constant_memory;
	unsigned long shared_memory_per_block;
	unsigned int max_thread_number_per_multiprocessor;
	unsigned int max_thread_number_per_block;
	unsigned int max_texture_dimension_1D;
	unsigned int max_texture_dimension_2D[2];
	unsigned int max_texture_dimension_3D[3];
	unsigned int dimension_size_of_thread[3];
	unsigned int dimension_size_of_grid[3];

}t_device_info;

extern t_device_info *deviceInfo;
void getGpuInfo();
void printDevInfo(t_device_info deviceInfo);

#endif /* GETDEVICEINFO_H_ */

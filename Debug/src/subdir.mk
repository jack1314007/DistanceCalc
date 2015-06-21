################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/distanceCalc.cu 

CPP_SRCS += \
../src/getDeviceInfo.cpp 

OBJS += \
./src/distanceCalc.o \
./src/getDeviceInfo.o 

CU_DEPS += \
./src/distanceCalc.d 

CPP_DEPS += \
./src/getDeviceInfo.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -G -g -O0 -Xcompiler -fPIC -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -G -g -O0 -Xcompiler -fPIC --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -G -g -O0 -Xcompiler -fPIC -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -G -g -O0 -Xcompiler -fPIC --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



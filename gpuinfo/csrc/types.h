#ifndef GPUINFO_TYPES_H
#define GPUINFO_TYPES_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>


typedef struct {
    PyObject_HEAD
    int ord;
    const char* provider;
    int index;
    const char* name;
    int major;
    int minor;
    size_t total_memory;
    int sms_count;
    int sm_threads;
    size_t sm_shared_memory;
    int sm_registers;
    int sm_blocks;
    int block_threads;
    size_t block_shared_memory;
    int block_registers;
    int warp_size;
    int l2_cache_size;
    char concurrent_kernels;
    int async_engines_count;
    char cooperative;

    char _provider_storage[8];
    char _name_storage[256];
} GpuProp;


int checkCuda();
int cudaGetDeviceCount(int* count);
int cudaGetDeviceProps(int index, GpuProp* obj);
void cudaClean();


int checkAmd();
int amdGetDeviceCount(int* count);
int amdGetDeviceProps(int index, GpuProp* obj);
void amdClean();


#endif //GPUINFO_TYPES_H

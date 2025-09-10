#ifndef GPUINFO_TYPES_H
#define GPUINFO_TYPES_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>


#define MAX_HINTS 32
#define MAX_HINT_LEN 127
#define HIN_OVEARHEAD 32

extern char _hints[MAX_HINTS][MAX_HINT_LEN + HIN_OVEARHEAD + 1];
extern int _hints_len[MAX_HINTS];
extern int _num_hints;


typedef struct {
    PyObject_HEAD
    int ord;
    const char* uuid;
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
    char _uuid_storage[32];
} GpuProp;


int checkCuda();
const char* cudaGetErrStr(int status);
int cudaGetDeviceCount(int* count);
int cudaGetDeviceProps(int index, GpuProp* obj);
void cudaClean();


int checkAmd();
const char* amdGetErrStr(int status);
int amdGetDeviceCount(int* count);
int amdGetDeviceProps(int index, GpuProp* obj);
void amdClean();


static inline void bytes_to_hex(const char* in, char* out, int bytes_len) {
    for (int i=0; i<bytes_len; ++i) {
        unsigned char high = ((const unsigned char*)(in))[i] >> 4;
        unsigned char low = ((const unsigned char*)(in))[i] & 0x0F;

        out[i*2] = (high < 10 ? '0' + high : 'a' + (high - 10));
        out[i*2+1] = (low < 10 ? '0' + low : 'a' + (low - 10));
    }
}


#endif //GPUINFO_TYPES_H

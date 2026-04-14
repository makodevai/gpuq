#ifndef GPUINFO_TYPES_H
#define GPUINFO_TYPES_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>


#ifndef FALSE
#define FALSE (0)
#endif

#ifndef TRUE
#define TRUE (1)
#endif


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

    char _provider_storage[8];
    char _name_storage[256];
    char _uuid_storage[32];
} GpuProp;


int checkCuda();
const char* cudaGetDlError();
int cudaGetDeviceCount(int* count);
int cudaGetDeviceProps(int index, GpuProp* obj);

/* NVML runtime info (utilisation, memory, PIDs) */
int nvmlGetRuntimeUtilisation(int index, int* gpu_util);
int nvmlGetRuntimeMemory(int index, unsigned long long* used_bytes);
int nvmlGetRuntimePids(int index, int* pids, int* count, int max_count);


int checkAmd();
const char* amdGetDlError();
const char* amdGetErrStr(int status);
int amdGetDeviceCount(int* count);
int amdGetDeviceProps(int index, GpuProp* obj);


// utils
void record_dl_error(const char** dl_error_buffer, size_t* dl_error_len, int append);


static inline void bytes_to_hex(const char* in, char* out, int bytes_len) {
    for (int i=0; i<bytes_len; ++i) {
        unsigned char high = ((const unsigned char*)(in))[i] >> 4;
        unsigned char low = ((const unsigned char*)(in))[i] & 0x0F;

        out[i*2] = (high < 10 ? '0' + high : 'a' + (high - 10));
        out[i*2+1] = (low < 10 ? '0' + low : 'a' + (low - 10));
    }
}


#endif //GPUINFO_TYPES_H

#include <stddef.h>
#include <string.h>
#include <dlfcn.h>

#include "types.h"


typedef void*        nvmlDevice_t;
typedef unsigned int nvmlReturn_t;

#define NVML_SUCCESS 0
#define NVML_DEVICE_NAME_BUFFER_SIZE 256
#define NVML_DEVICE_UUID_BUFFER_SIZE 80
#define NVML_MAX_PROCS 128

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef struct {
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
} nvmlProcessInfo_t;

typedef nvmlReturn_t (*nvmlInit_v2_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetCount_v2_t)(unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetName_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetUUID_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*nvmlDeviceGetCudaComputeCapability_t)(nvmlDevice_t, int*, int*);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t (*nvmlDeviceGetComputeRunningProcesses_v3_t)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*);


/* ── CUDA Driver API attribute IDs ───────────────────────────────────── */

#define CU_ATTR_MAX_THREADS_PER_BLOCK              1
#define CU_ATTR_MAX_SHARED_MEMORY_PER_BLOCK        8
#define CU_ATTR_WARP_SIZE                          10
#define CU_ATTR_MAX_REGISTERS_PER_BLOCK            12
#define CU_ATTR_MULTIPROCESSOR_COUNT               16
#define CU_ATTR_CONCURRENT_KERNELS                 31
#define CU_ATTR_L2_CACHE_SIZE                      38
#define CU_ATTR_MAX_THREADS_PER_MULTIPROCESSOR     39
#define CU_ATTR_ASYNC_ENGINE_COUNT                 40
#define CU_ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR 81
#define CU_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR   82
#define CU_ATTR_COOPERATIVE_LAUNCH                 95
#define CU_ATTR_MAX_BLOCKS_PER_MULTIPROCESSOR     106

#define CUDA_SUCCESS 0
#define MAX_CUDA_DEVICES 32

typedef int CUresult;
typedef int CUdevice;

typedef struct { char bytes[16]; } CUuuid;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGetCount_t)(int*);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetAttribute_t)(int*, int, CUdevice);
typedef CUresult (*cuDeviceGetUuid_t)(CUuuid*, CUdevice);


static void try_load_cuda_driver(void);

static const char* dl_error_buffer = NULL;
static size_t dl_error_len = 0;

/* ── NVML state ─────────────────────────────────────────────────────── */

static void* nvml_dl                    = NULL;
static nvmlInit_v2_t                    nvml_init_fn   = NULL;
static nvmlDeviceGetCount_v2_t          nvml_count_fn  = NULL;
static nvmlDeviceGetHandleByIndex_v2_t  nvml_handle_fn = NULL;
static nvmlDeviceGetName_t              nvml_name_fn   = NULL;
static nvmlDeviceGetUUID_t              nvml_uuid_fn   = NULL;
static nvmlDeviceGetMemoryInfo_t        nvml_mem_fn    = NULL;
static nvmlDeviceGetCudaComputeCapability_t nvml_cc_fn = NULL;
static nvmlDeviceGetUtilizationRates_t  nvml_util_fn   = NULL;
static nvmlDeviceGetComputeRunningProcesses_v3_t nvml_procs_fn = NULL;


static int try_load_nvml() {
    if (nvml_dl) return 0;  /* already loaded */

    nvml_dl = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!nvml_dl) {
        record_dl_error(&dl_error_buffer, &dl_error_len, FALSE);
        nvml_dl = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!nvml_dl) {
        record_dl_error(&dl_error_buffer, &dl_error_len, TRUE);
        return -1;
    }

#define LOAD_SYM(var, sym)                                      \
    var = (typeof(var))dlsym(nvml_dl, #sym);                    \
    if (!var) { record_dl_error(&dl_error_buffer, &dl_error_len, FALSE); dlclose(nvml_dl); nvml_dl = NULL; return -1; }

    LOAD_SYM(nvml_init_fn,   nvmlInit_v2)
    LOAD_SYM(nvml_count_fn,  nvmlDeviceGetCount_v2)
    LOAD_SYM(nvml_handle_fn, nvmlDeviceGetHandleByIndex_v2)
    LOAD_SYM(nvml_name_fn,   nvmlDeviceGetName)
    LOAD_SYM(nvml_uuid_fn,   nvmlDeviceGetUUID)
    LOAD_SYM(nvml_mem_fn,    nvmlDeviceGetMemoryInfo)
    LOAD_SYM(nvml_cc_fn,     nvmlDeviceGetCudaComputeCapability)
#undef LOAD_SYM

    /* optional symbols - not fatal if missing */
    nvml_util_fn = (nvmlDeviceGetUtilizationRates_t)dlsym(nvml_dl, "nvmlDeviceGetUtilizationRates");
    nvml_procs_fn = (nvmlDeviceGetComputeRunningProcesses_v3_t)dlsym(nvml_dl, "nvmlDeviceGetComputeRunningProcesses_v3");

    if (nvml_init_fn() != NVML_SUCCESS) {
        dlclose(nvml_dl);
        nvml_dl = NULL;
        return -1;
    }

    if (dl_error_buffer) {
        free((void*)dl_error_buffer);
        dl_error_buffer = NULL;
        dl_error_len = 0;
    }

    try_load_cuda_driver();  /* optional, soft-fail */
    return 0;
}


/* ── CUDA Driver API (optional, for device attributes) ──────────────── */

static void* cuda_dl = NULL;
static cuInit_t             cu_init_fn  = NULL;
static cuDeviceGetCount_t   cu_count_fn = NULL;
static cuDeviceGet_t        cu_get_fn   = NULL;
static cuDeviceGetAttribute_t cu_attr_fn = NULL;
static cuDeviceGetUuid_t    cu_uuid_fn  = NULL;

static CUdevice cu_dev_handles[MAX_CUDA_DEVICES];
static char     cu_dev_uuids[MAX_CUDA_DEVICES][33];  /* 32 hex + nul */
static int      cu_dev_count = 0;
static int      cuda_driver_available = 0;


static void try_load_cuda_driver() {
    if (cuda_dl) return;

    cuda_dl = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!cuda_dl)
        cuda_dl = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    if (!cuda_dl) return;

    cu_init_fn  = (cuInit_t)dlsym(cuda_dl, "cuInit");
    cu_count_fn = (cuDeviceGetCount_t)dlsym(cuda_dl, "cuDeviceGetCount");
    cu_get_fn   = (cuDeviceGet_t)dlsym(cuda_dl, "cuDeviceGet");
    cu_attr_fn  = (cuDeviceGetAttribute_t)dlsym(cuda_dl, "cuDeviceGetAttribute");
    cu_uuid_fn  = (cuDeviceGetUuid_t)dlsym(cuda_dl, "cuDeviceGetUuid");

    if (!cu_init_fn || !cu_count_fn || !cu_get_fn || !cu_attr_fn || !cu_uuid_fn) {
        dlclose(cuda_dl); cuda_dl = NULL; return;
    }

    if (cu_init_fn(0) != CUDA_SUCCESS) {
        dlclose(cuda_dl); cuda_dl = NULL; return;
    }

    int count = 0;
    if (cu_count_fn(&count) != CUDA_SUCCESS || count <= 0) {
        cuda_driver_available = 1;  /* driver works, just no visible devices */
        return;
    }

    if (count > MAX_CUDA_DEVICES) count = MAX_CUDA_DEVICES;
    for (int i = 0; i < count; i++) {
        CUdevice dev;
        if (cu_get_fn(&dev, i) != CUDA_SUCCESS) continue;

        CUuuid uuid;
        if (cu_uuid_fn(&uuid, dev) != CUDA_SUCCESS) continue;

        cu_dev_handles[cu_dev_count] = dev;
        bytes_to_hex(uuid.bytes, cu_dev_uuids[cu_dev_count], 16);
        cu_dev_uuids[cu_dev_count][32] = '\0';
        cu_dev_count++;
    }

    cuda_driver_available = 1;
}


static int find_cuda_device_by_uuid(const char* uuid_hex) {
    for (int i = 0; i < cu_dev_count; i++) {
        if (memcmp(cu_dev_uuids[i], uuid_hex, 32) == 0)
            return cu_dev_handles[i];
    }
    return -1;
}


/* Convert NVML UUID string "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" to
   32-char hex (no prefix, no dashes). */
static void nvml_uuid_to_hex(const char* src, char* out) {
    if (strncmp(src, "GPU-", 4) == 0)
        src += 4;

    int out_idx = 0;
    for (int i = 0; src[i] != '\0' && out_idx < 32; i++) {
        if (src[i] != '-')
            out[out_idx++] = src[i];
    }
}


static int get_handle(int index, nvmlDevice_t* handle) {
    if (try_load_nvml() != 0) return -1;
    if (nvml_handle_fn((unsigned int)index, handle) != NVML_SUCCESS) return -1;
    return 0;
}


int checkCuda() {
    return try_load_nvml();
}


const char* cudaGetDlError() {
    return dl_error_buffer;
}


int cudaGetDeviceCount(int* count) {
    if (try_load_nvml() != 0) return -1;
    unsigned int n = 0;
    if (nvml_count_fn(&n) != NVML_SUCCESS) return -1;
    *count = (int)n;
    return 0;
}


int cudaGetDeviceProps(int index, GpuProp* obj) {
    nvmlDevice_t handle = NULL;
    if (get_handle(index, &handle) != 0) return -1;

    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {0};
    if (nvml_name_fn(handle, name, sizeof(name)) == NVML_SUCCESS)
        memcpy(obj->_name_storage, name, 256);

    char uuid_str[NVML_DEVICE_UUID_BUFFER_SIZE] = {0};
    if (nvml_uuid_fn(handle, uuid_str, sizeof(uuid_str)) == NVML_SUCCESS)
        nvml_uuid_to_hex(uuid_str, obj->_uuid_storage);

    nvmlMemory_t mem = {0, 0, 0};
    if (nvml_mem_fn(handle, &mem) == NVML_SUCCESS)
        obj->total_memory = mem.total;

    int major = 0, minor = 0;
    if (nvml_cc_fn(handle, &major, &minor) == NVML_SUCCESS) {
        obj->major = major;
        obj->minor = minor;
    }

    strcpy(obj->_provider_storage, "CUDA");
    obj->index = index;

    /* zero all driver-API fields by default */
    obj->sms_count = 0;
    obj->l2_cache_size = 0;
    obj->sm_threads = 0;
    obj->sm_shared_memory = 0;
    obj->sm_registers = 0;
    obj->sm_blocks = 0;
    obj->block_threads = 0;
    obj->block_shared_memory = 0;
    obj->block_registers = 0;
    obj->warp_size = 0;
    obj->concurrent_kernels = 0;
    obj->async_engines_count = 0;
    obj->cooperative = 0;

    /* populate from CUDA Driver API if available */
    if (cuda_driver_available) {
        int cu_dev = find_cuda_device_by_uuid(obj->_uuid_storage);
        if (cu_dev >= 0) {
            int val = 0;

#define QUERY_ATTR(field, attr) \
    if (cu_attr_fn(&val, attr, cu_dev) == CUDA_SUCCESS) obj->field = val;

            QUERY_ATTR(sms_count,          CU_ATTR_MULTIPROCESSOR_COUNT)
            QUERY_ATTR(l2_cache_size,      CU_ATTR_L2_CACHE_SIZE)
            QUERY_ATTR(sm_threads,         CU_ATTR_MAX_THREADS_PER_MULTIPROCESSOR)
            QUERY_ATTR(sm_registers,       CU_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR)
            QUERY_ATTR(sm_blocks,          CU_ATTR_MAX_BLOCKS_PER_MULTIPROCESSOR)
            QUERY_ATTR(block_threads,      CU_ATTR_MAX_THREADS_PER_BLOCK)
            QUERY_ATTR(block_registers,    CU_ATTR_MAX_REGISTERS_PER_BLOCK)
            QUERY_ATTR(warp_size,          CU_ATTR_WARP_SIZE)
            QUERY_ATTR(async_engines_count, CU_ATTR_ASYNC_ENGINE_COUNT)

            if (cu_attr_fn(&val, CU_ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cu_dev) == CUDA_SUCCESS)
                obj->sm_shared_memory = (size_t)val;
            if (cu_attr_fn(&val, CU_ATTR_MAX_SHARED_MEMORY_PER_BLOCK, cu_dev) == CUDA_SUCCESS)
                obj->block_shared_memory = (size_t)val;
            if (cu_attr_fn(&val, CU_ATTR_CONCURRENT_KERNELS, cu_dev) == CUDA_SUCCESS)
                obj->concurrent_kernels = (char)(val != 0);
            if (cu_attr_fn(&val, CU_ATTR_COOPERATIVE_LAUNCH, cu_dev) == CUDA_SUCCESS)
                obj->cooperative = (char)(val != 0);

#undef QUERY_ATTR
        }
    }

    return 0;
}


/* ── runtime info (utilisation, memory, PIDs) ────────────────────────── */

int nvmlGetRuntimeUtilisation(int index, int* gpu_util) {
    nvmlDevice_t handle = NULL;
    if (get_handle(index, &handle) != 0) return -1;
    if (!nvml_util_fn) return -1;

    nvmlUtilization_t util = {0, 0};
    if (nvml_util_fn(handle, &util) != NVML_SUCCESS) return -1;
    *gpu_util = (int)util.gpu;
    return 0;
}


int nvmlGetRuntimeMemory(int index, unsigned long long* used_bytes) {
    nvmlDevice_t handle = NULL;
    if (get_handle(index, &handle) != 0) return -1;

    nvmlMemory_t mem = {0, 0, 0};
    if (nvml_mem_fn(handle, &mem) != NVML_SUCCESS) return -1;
    *used_bytes = mem.used;
    return 0;
}


int nvmlGetRuntimePids(int index, int* pids, int* count, int max_count) {
    nvmlDevice_t handle = NULL;
    if (get_handle(index, &handle) != 0) { *count = 0; return -1; }
    if (!nvml_procs_fn) { *count = 0; return -1; }

    unsigned int info_count = NVML_MAX_PROCS;
    nvmlProcessInfo_t infos[NVML_MAX_PROCS];
    memset(infos, 0, sizeof(infos));

    if (nvml_procs_fn(handle, &info_count, infos) != NVML_SUCCESS) {
        *count = 0;
        return -1;
    }

    int n = 0;
    for (unsigned int i = 0; i < info_count && n < max_count; i++) {
        pids[n++] = (int)infos[i].pid;
    }
    *count = n;
    return 0;
}

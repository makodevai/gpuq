#include <stddef.h>
#include <string.h>
#include <dlfcn.h>

#include "types.h"

typedef void*        nvmlDevice_t;
typedef unsigned int nvmlReturn_t;

#define NVML_SUCCESS 0
#define NVML_DEVICE_NAME_BUFFER_SIZE 256
#define NVML_DEVICE_UUID_BUFFER_SIZE 80

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef nvmlReturn_t (*nvmlInit_v2_t)(void);
typedef nvmlReturn_t (*nvmlDeviceGetCount_v2_t)(unsigned int*);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2_t)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetName_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetUUID_t)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*nvmlDeviceGetCudaComputeCapability_t)(nvmlDevice_t, int*, int*);


static void* nvml_dl                    = NULL;
static nvmlInit_v2_t                    nvml_init_fn   = NULL;
static nvmlDeviceGetCount_v2_t          nvml_count_fn  = NULL;
static nvmlDeviceGetHandleByIndex_v2_t  nvml_handle_fn = NULL;
static nvmlDeviceGetName_t              nvml_name_fn   = NULL;
static nvmlDeviceGetUUID_t              nvml_uuid_fn   = NULL;
static nvmlDeviceGetMemoryInfo_t        nvml_mem_fn    = NULL;
static nvmlDeviceGetCudaComputeCapability_t nvml_cc_fn = NULL;


static int try_load_nvml() {
    if (nvml_dl) return 0;  /* already loaded */

    nvml_dl = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!nvml_dl)
        nvml_dl = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
    if (!nvml_dl)
        return -1;

#define LOAD_SYM(var, sym)                                      \
    var = (typeof(var))dlsym(nvml_dl, #sym);                    \
    if (!var) { dlclose(nvml_dl); nvml_dl = NULL; return -1; }

    LOAD_SYM(nvml_init_fn,   nvmlInit_v2)
    LOAD_SYM(nvml_count_fn,  nvmlDeviceGetCount_v2)
    LOAD_SYM(nvml_handle_fn, nvmlDeviceGetHandleByIndex_v2)
    LOAD_SYM(nvml_name_fn,   nvmlDeviceGetName)
    LOAD_SYM(nvml_uuid_fn,   nvmlDeviceGetUUID)
    LOAD_SYM(nvml_mem_fn,    nvmlDeviceGetMemoryInfo)
    LOAD_SYM(nvml_cc_fn,     nvmlDeviceGetCudaComputeCapability)
#undef LOAD_SYM

    if (nvml_init_fn() != NVML_SUCCESS) {
        dlclose(nvml_dl);
        nvml_dl = NULL;
        return -1;
    }
    return 0;
}


/* Convert NVML UUID string "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" to the
   same 32-char hex format used by bytes_to_hex() for CUDA UUIDs (no prefix,
   no dashes). The output buffer must be at least 32 bytes; no null terminator
   is written (matches the existing CUDA convention in _uuid_storage[32]). */
static void nvml_uuid_to_hex(const char* src, char* out) {
    /* skip "GPU-" prefix if present */
    if (strncmp(src, "GPU-", 4) == 0)
        src += 4;

    int out_idx = 0;
    for (int i = 0; src[i] != '\0' && out_idx < 32; i++) {
        if (src[i] != '-')
            out[out_idx++] = src[i];
    }
}


int nvmlGetPhysicalDeviceCount(int* count) {
    if (try_load_nvml() != 0) return -1;
    unsigned int n = 0;
    if (nvml_count_fn(&n) != NVML_SUCCESS) return -1;
    *count = (int)n;
    return 0;
}


int nvmlGetDeviceProps(int index, GpuProp* obj) {
    if (try_load_nvml() != 0) return -1;

    nvmlDevice_t handle = NULL;
    if (nvml_handle_fn((unsigned int)index, &handle) != NVML_SUCCESS) return -1;

    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {0};
    nvml_name_fn(handle, name, sizeof(name));
    memcpy(obj->_name_storage, name, 256);

    char uuid_str[NVML_DEVICE_UUID_BUFFER_SIZE] = {0};
    nvml_uuid_fn(handle, uuid_str, sizeof(uuid_str));
    nvml_uuid_to_hex(uuid_str, obj->_uuid_storage);

    nvmlMemory_t mem = {0, 0, 0};
    nvml_mem_fn(handle, &mem);
    obj->total_memory = mem.total;

    int major = 0, minor = 0;
    nvml_cc_fn(handle, &major, &minor);
    obj->major = major;
    obj->minor = minor;

    strcpy(obj->_provider_storage, "CUDA");
    obj->index = index;
    obj->sms_count          = 0;
    obj->sm_threads         = 0;
    obj->sm_shared_memory   = 0;
    obj->sm_registers       = 0;
    obj->sm_blocks          = 0;
    obj->block_threads      = 0;
    obj->block_shared_memory = 0;
    obj->block_registers    = 0;
    obj->warp_size          = 32;
    obj->l2_cache_size      = 0;
    obj->concurrent_kernels = 0;
    obj->async_engines_count = 0;
    obj->cooperative        = 0;

    return 0;
}

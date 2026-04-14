/* AMD GPU support via AMD SMI (replaces former libamdhip64 binding). */

#include <stddef.h>
#include <string.h>
#include <dlfcn.h>

#include "types.h"


typedef void*    amdsmi_processor_handle;
typedef void*    amdsmi_socket_handle;
typedef uint32_t amdsmi_status_t;
typedef uint32_t amdsmi_process_handle_t;

#define AMDSMI_STATUS_SUCCESS       0
#define AMDSMI_INIT_AMD_GPUS        (1 << 1)
#define AMDSMI_MAX_STRING_LENGTH    256
#define AMDSMI_MAX_CACHE_TYPES      10
#define AMDSMI_MAX_DEVICES          32
#define AMDSMI_MAX_PROCS            128
#define AMDSMI_MEM_TYPE_VRAM        0

typedef struct {
    char     market_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t vendor_id;
    char     vendor_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t subvendor_id;
    uint64_t device_id;
    uint32_t rev_id;
    char     asic_serial[AMDSMI_MAX_STRING_LENGTH];
    uint32_t oam_id;
    uint32_t num_of_compute_units;
    uint64_t target_graphics_version;
    uint32_t subsystem_id;
    uint32_t reserved[21];
} amdsmi_asic_info_t;

typedef struct {
    uint32_t num_cache_types;
    struct {
        uint32_t cache_properties;
        uint32_t cache_size;       /* in KB */
        uint32_t cache_level;
        uint32_t max_num_cu_shared;
        uint32_t num_cache_instance;
        uint32_t reserved[3];
    } cache[AMDSMI_MAX_CACHE_TYPES];
    uint32_t reserved[15];
} amdsmi_gpu_cache_info_t;

typedef struct {
    uint32_t gfx_activity;
    uint32_t umc_activity;
    uint32_t mm_activity;
    uint32_t reserved[13];
} amdsmi_engine_usage_t;

typedef struct {
    char name[AMDSMI_MAX_STRING_LENGTH];
    amdsmi_process_handle_t pid;
    uint64_t mem;
    struct {
        uint64_t gfx;
        uint64_t enc;
        uint32_t reserved[12];
    } engine_usage;
    struct {
        uint64_t gtt_mem;
        uint64_t cpu_mem;
        uint64_t vram_mem;
        uint32_t reserved[10];
    } memory_usage;
    char container_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t cu_occupancy;
    uint32_t evicted_time;
    uint32_t reserved[10];
} amdsmi_proc_info_t;

typedef struct {
    uint32_t drm_render;
    uint32_t drm_card;
    uint32_t hsa_id;
    uint32_t hip_id;
    char hip_uuid[AMDSMI_MAX_STRING_LENGTH];
} amdsmi_enumeration_info_t;

typedef struct {
    uint64_t kfd_id;
    uint32_t node_id;
    uint32_t current_partition_id;
    uint32_t reserved[12];
} amdsmi_kfd_info_t;


typedef amdsmi_status_t (*amdsmi_init_t)(uint64_t);
typedef amdsmi_status_t (*amdsmi_shut_down_t)(void);
typedef amdsmi_status_t (*amdsmi_get_socket_handles_t)(uint32_t*, amdsmi_socket_handle*);
typedef amdsmi_status_t (*amdsmi_get_processor_handles_t)(amdsmi_socket_handle, uint32_t*, amdsmi_processor_handle*);
typedef amdsmi_status_t (*amdsmi_get_gpu_asic_info_t)(amdsmi_processor_handle, amdsmi_asic_info_t*);
typedef amdsmi_status_t (*amdsmi_get_gpu_cache_info_t)(amdsmi_processor_handle, amdsmi_gpu_cache_info_t*);
typedef amdsmi_status_t (*amdsmi_get_gpu_device_uuid_t)(amdsmi_processor_handle, unsigned int*, char*);
typedef amdsmi_status_t (*amdsmi_get_gpu_memory_total_t)(amdsmi_processor_handle, uint32_t, uint64_t*);
typedef amdsmi_status_t (*amdsmi_get_gpu_memory_usage_t)(amdsmi_processor_handle, uint32_t, uint64_t*);
typedef amdsmi_status_t (*amdsmi_get_gpu_activity_t)(amdsmi_processor_handle, amdsmi_engine_usage_t*);
typedef amdsmi_status_t (*amdsmi_get_gpu_process_list_t)(amdsmi_processor_handle, uint32_t*, amdsmi_proc_info_t*);
typedef amdsmi_status_t (*amdsmi_get_gpu_enumeration_info_t)(amdsmi_processor_handle, amdsmi_enumeration_info_t*);
typedef amdsmi_status_t (*amdsmi_get_gpu_kfd_info_t)(amdsmi_processor_handle, amdsmi_kfd_info_t*);


static const char* dl_error_buffer = NULL;
static size_t dl_error_len = 0;

static void* amdsmi_dl = NULL;

static amdsmi_init_t                    smi_init_fn       = NULL;
static amdsmi_get_socket_handles_t      smi_sockets_fn    = NULL;
static amdsmi_get_processor_handles_t   smi_procs_fn      = NULL;
static amdsmi_get_gpu_asic_info_t       smi_asic_fn       = NULL;
static amdsmi_get_gpu_cache_info_t      smi_cache_fn      = NULL;
static amdsmi_get_gpu_device_uuid_t     smi_uuid_fn       = NULL;
static amdsmi_get_gpu_memory_total_t    smi_mem_total_fn   = NULL;
static amdsmi_get_gpu_memory_usage_t    smi_mem_usage_fn   = NULL;
static amdsmi_get_gpu_activity_t        smi_activity_fn    = NULL;
static amdsmi_get_gpu_process_list_t    smi_proc_list_fn   = NULL;
static amdsmi_get_gpu_enumeration_info_t smi_enum_fn       = NULL;
static amdsmi_get_gpu_kfd_info_t        smi_kfd_fn         = NULL;

/* flat list of GPU handles, populated at init */
static amdsmi_processor_handle gpu_handles[AMDSMI_MAX_DEVICES];
static int gpu_count = 0;


static int try_load_amdsmi() {
    if (amdsmi_dl) return 0;

    amdsmi_dl = dlopen("libamd_smi.so", RTLD_NOW | RTLD_LOCAL);
    if (!amdsmi_dl) {
        record_dl_error(&dl_error_buffer, &dl_error_len, FALSE);
        amdsmi_dl = dlopen("/opt/rocm/lib/libamd_smi.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!amdsmi_dl) {
        record_dl_error(&dl_error_buffer, &dl_error_len, TRUE);
        return -1;
    }

#define LOAD_SYM(var, sym)                                                  \
    var = (typeof(var))dlsym(amdsmi_dl, #sym);                              \
    if (!var) { record_dl_error(&dl_error_buffer, &dl_error_len, FALSE);    \
                dlclose(amdsmi_dl); amdsmi_dl = NULL; return -1; }

    LOAD_SYM(smi_init_fn,       amdsmi_init)
    LOAD_SYM(smi_sockets_fn,    amdsmi_get_socket_handles)
    LOAD_SYM(smi_procs_fn,      amdsmi_get_processor_handles)
    LOAD_SYM(smi_asic_fn,       amdsmi_get_gpu_asic_info)
    LOAD_SYM(smi_uuid_fn,       amdsmi_get_gpu_device_uuid)
    LOAD_SYM(smi_mem_total_fn,  amdsmi_get_gpu_memory_total)
#undef LOAD_SYM

    /* optional symbols — not fatal if missing */
    smi_cache_fn     = (amdsmi_get_gpu_cache_info_t)dlsym(amdsmi_dl, "amdsmi_get_gpu_cache_info");
    smi_mem_usage_fn = (amdsmi_get_gpu_memory_usage_t)dlsym(amdsmi_dl, "amdsmi_get_gpu_memory_usage");
    smi_activity_fn  = (amdsmi_get_gpu_activity_t)dlsym(amdsmi_dl, "amdsmi_get_gpu_activity");
    smi_proc_list_fn = (amdsmi_get_gpu_process_list_t)dlsym(amdsmi_dl, "amdsmi_get_gpu_process_list");
    smi_enum_fn      = (amdsmi_get_gpu_enumeration_info_t)dlsym(amdsmi_dl, "amdsmi_get_gpu_enumeration_info");
    smi_kfd_fn       = (amdsmi_get_gpu_kfd_info_t)dlsym(amdsmi_dl, "amdsmi_get_gpu_kfd_info");

    if (smi_init_fn(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS) {
        dlclose(amdsmi_dl);
        amdsmi_dl = NULL;
        return -1;
    }

    /* enumerate all GPU handles into a flat array */
    uint32_t sock_count = 0;
    if (smi_sockets_fn(&sock_count, NULL) != AMDSMI_STATUS_SUCCESS || sock_count == 0) {
        gpu_count = 0;
        goto done;
    }

    amdsmi_socket_handle sockets[32];
    if (sock_count > 32) sock_count = 32;
    if (smi_sockets_fn(&sock_count, sockets) != AMDSMI_STATUS_SUCCESS) {
        gpu_count = 0;
        goto done;
    }

    gpu_count = 0;
    for (uint32_t s = 0; s < sock_count && gpu_count < AMDSMI_MAX_DEVICES; s++) {
        uint32_t proc_count = 0;
        if (smi_procs_fn(sockets[s], &proc_count, NULL) != AMDSMI_STATUS_SUCCESS)
            continue;

        amdsmi_processor_handle procs[AMDSMI_MAX_DEVICES];
        if (proc_count > AMDSMI_MAX_DEVICES) proc_count = AMDSMI_MAX_DEVICES;
        if (smi_procs_fn(sockets[s], &proc_count, procs) != AMDSMI_STATUS_SUCCESS)
            continue;

        for (uint32_t p = 0; p < proc_count && gpu_count < AMDSMI_MAX_DEVICES; p++) {
            gpu_handles[gpu_count++] = procs[p];
        }
    }

done:
    if (dl_error_buffer) {
        free((void*)dl_error_buffer);
        dl_error_buffer = NULL;
        dl_error_len = 0;
    }
    return 0;
}


int checkAmd() {
    return try_load_amdsmi();
}


const char* amdGetDlError() {
    return dl_error_buffer;
}


int amdGetDeviceCount(int* count) {
    if (try_load_amdsmi() != 0) return -1;
    *count = gpu_count;
    return 0;
}


int amdGetDeviceProps(int index, GpuProp* obj) {
    if (try_load_amdsmi() != 0) return -1;
    if (index < 0 || index >= gpu_count) return -1;

    amdsmi_processor_handle handle = gpu_handles[index];

    /* name and compute units via asic info */
    amdsmi_asic_info_t asic = {0};
    if (smi_asic_fn(handle, &asic) == AMDSMI_STATUS_SUCCESS) {
        strncpy(obj->_name_storage, asic.market_name, 255);
        obj->_name_storage[255] = '\0';
        obj->sms_count = (asic.num_of_compute_units != 0xFFFFFFFF)
            ? (int)asic.num_of_compute_units : 0;
    }

    /* UUID */
    char uuid_buf[256] = {0};
    unsigned int uuid_len = sizeof(uuid_buf);
    if (smi_uuid_fn(handle, &uuid_len, uuid_buf) == AMDSMI_STATUS_SUCCESS) {
        /* AMD SMI returns a string UUID; strip any prefix/dashes into 32 hex chars */
        const char* src = uuid_buf;
        int out_idx = 0;
        for (int i = 0; src[i] != '\0' && out_idx < 32; i++) {
            if (src[i] != '-' && src[i] != ' ')
                obj->_uuid_storage[out_idx++] = src[i];
        }
    }

    /* total memory in bytes */
    uint64_t mem_total = 0;
    if (smi_mem_total_fn(handle, AMDSMI_MEM_TYPE_VRAM, &mem_total) == AMDSMI_STATUS_SUCCESS)
        obj->total_memory = (size_t)mem_total;

    /* L2 cache */
    obj->l2_cache_size = 0;
    if (smi_cache_fn) {
        amdsmi_gpu_cache_info_t cache_info = {0};
        if (smi_cache_fn(handle, &cache_info) == AMDSMI_STATUS_SUCCESS) {
            for (uint32_t i = 0; i < cache_info.num_cache_types && i < AMDSMI_MAX_CACHE_TYPES; i++) {
                if (cache_info.cache[i].cache_level == 2) {
                    obj->l2_cache_size = (int)cache_info.cache[i].cache_size; /* in KB */
                    break;
                }
            }
        }
    }

    strcpy(obj->_provider_storage, "HIP");
    obj->index = index;
    obj->major = 0;
    obj->minor = 0;

    return 0;
}


/* ── runtime info ────────────────────────────────────────────────────── */

int amdsmiGetRuntimeUtilisation(int index, int* gpu_util) {
    if (try_load_amdsmi() != 0 || index < 0 || index >= gpu_count) return -1;
    if (!smi_activity_fn) return -1;

    amdsmi_engine_usage_t usage = {0};
    if (smi_activity_fn(gpu_handles[index], &usage) != AMDSMI_STATUS_SUCCESS) return -1;
    *gpu_util = (int)usage.gfx_activity;
    return 0;
}


int amdsmiGetRuntimeMemory(int index, unsigned long long* used_bytes) {
    if (try_load_amdsmi() != 0 || index < 0 || index >= gpu_count) return -1;
    if (!smi_mem_usage_fn) return -1;

    uint64_t used = 0;
    if (smi_mem_usage_fn(gpu_handles[index], AMDSMI_MEM_TYPE_VRAM, &used) != AMDSMI_STATUS_SUCCESS) return -1;
    *used_bytes = used;
    return 0;
}


int amdsmiGetRuntimePids(int index, int* pids, int* count, int max_count) {
    *count = 0;
    if (try_load_amdsmi() != 0 || index < 0 || index >= gpu_count) return -1;
    if (!smi_proc_list_fn) return -1;

    /* two-call pattern: get count first */
    uint32_t num_procs = 0;
    smi_proc_list_fn(gpu_handles[index], &num_procs, NULL);
    if (num_procs == 0) return 0;

    if (num_procs > AMDSMI_MAX_PROCS) num_procs = AMDSMI_MAX_PROCS;
    amdsmi_proc_info_t infos[AMDSMI_MAX_PROCS];
    memset(infos, 0, sizeof(infos));

    if (smi_proc_list_fn(gpu_handles[index], &num_procs, infos) != AMDSMI_STATUS_SUCCESS)
        return -1;

    int n = 0;
    for (uint32_t i = 0; i < num_procs && n < max_count; i++) {
        pids[n++] = (int)infos[i].pid;
    }
    *count = n;
    return 0;
}


int amdsmiGetGfxVersion(int index, char* gfx, int max_len) {
    if (try_load_amdsmi() != 0 || index < 0 || index >= gpu_count) return -1;

    amdsmi_asic_info_t asic = {0};
    if (smi_asic_fn(gpu_handles[index], &asic) != AMDSMI_STATUS_SUCCESS) return -1;

    if (asic.target_graphics_version == 0xFFFFFFFFFFFFFFFFULL) return -1;

    /* encode as "major.minor.stepping" decimal string */
    uint64_t v = asic.target_graphics_version;
    snprintf(gfx, max_len, "%u.%u.%u",
             (unsigned)((v >> 24) & 0xFF),
             (unsigned)((v >> 16) & 0xFF),
             (unsigned)(v & 0xFFFF));
    return 0;
}


int amdsmiGetDrmRender(int index, int* drm_render) {
    if (try_load_amdsmi() != 0 || index < 0 || index >= gpu_count) return -1;
    if (!smi_enum_fn) return -1;

    amdsmi_enumeration_info_t info = {0};
    if (smi_enum_fn(gpu_handles[index], &info) != AMDSMI_STATUS_SUCCESS) return -1;
    *drm_render = (int)info.drm_render;
    return 0;
}


int amdsmiGetNodeId(int index, int* node_id) {
    if (try_load_amdsmi() != 0 || index < 0 || index >= gpu_count) return -1;
    if (!smi_kfd_fn) return -1;

    amdsmi_kfd_info_t info = {0};
    if (smi_kfd_fn(gpu_handles[index], &info) != AMDSMI_STATUS_SUCCESS) return -1;
    if (info.node_id == 0xFFFFFFFF) return -1;
    *node_id = (int)info.node_id;
    return 0;
}

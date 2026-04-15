/* AMD GPU support via AMD SMI + HSA Runtime.
   Type definitions below are copied from AMD SMI / HSA headers (ROCm 7.2.0)
   to avoid a build-time dependency on the ROCm SDK.  May need updating if
   AMD changes struct layouts in a future ROCm release. */

#include <stddef.h>
#include <string.h>
#include <dlfcn.h>

#include "types.h"


/* Opaque handles */
typedef void*    amdsmi_processor_handle;  /* per-GPU handle */
typedef void*    amdsmi_socket_handle;     /* per-socket (physical package) handle */
typedef uint32_t amdsmi_status_t;         /* return code, 0 = success */
typedef uint32_t amdsmi_process_handle_t; /* PID type for process queries */

#define AMDSMI_STATUS_SUCCESS       0
#define AMDSMI_INIT_AMD_GPUS        (1 << 1)  /* init flag: GPUs only, skip CPUs */
#define AMDSMI_MAX_STRING_LENGTH    256
#define AMDSMI_MAX_CACHE_TYPES      10
#define AMDSMI_MAX_DEVICES          32
#define AMDSMI_MAX_PROCS            128
#define AMDSMI_MEM_TYPE_VRAM        0          /* memory type selector for VRAM queries */

/* GPU ASIC info — name, compute units, gfx version, etc. */
typedef struct {
    char     market_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t vendor_id;
    char     vendor_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t subvendor_id;
    uint64_t device_id;
    uint32_t rev_id;
    char     asic_serial[AMDSMI_MAX_STRING_LENGTH];
    uint32_t oam_id;
    uint32_t num_of_compute_units;      /* CU count, 0xFFFFFFFF = unknown */
    uint64_t target_graphics_version;   /* gfx version as hex, e.g. 0x950 = gfx950 */
    uint32_t subsystem_id;
    uint32_t reserved[21];
} amdsmi_asic_info_t;

/* GPU cache hierarchy info */
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

/* GPU engine utilisation — gfx, memory controller, multimedia */
typedef struct {
    uint32_t gfx_activity;   /* graphics engine busy, 0-100 % */
    uint32_t umc_activity;   /* memory controller busy, 0-100 % */
    uint32_t mm_activity;    /* multimedia engine busy, 0-100 % */
    uint32_t reserved[13];
} amdsmi_engine_usage_t;

/* Per-process GPU usage info */
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
        uint64_t gtt_mem;   /* system memory mapped for GPU */
        uint64_t cpu_mem;
        uint64_t vram_mem;  /* GPU VRAM used by this process */
        uint32_t reserved[10];
    } memory_usage;
    char container_name[AMDSMI_MAX_STRING_LENGTH];
    uint32_t cu_occupancy;
    uint32_t evicted_time;
    uint32_t reserved[10];
} amdsmi_proc_info_t;

/* Device enumeration — DRM render node, HIP/HSA IDs, UUID */
typedef struct {
    uint32_t drm_render;  /* /dev/dri/renderDN minor number */
    uint32_t drm_card;    /* /dev/dri/cardN minor number */
    uint32_t hsa_id;
    uint32_t hip_id;
    char hip_uuid[AMDSMI_MAX_STRING_LENGTH];
} amdsmi_enumeration_info_t;

/* KFD (kernel fusion driver) info — node topology IDs */
typedef struct {
    uint64_t kfd_id;
    uint32_t node_id;              /* KFD node index */
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


/* ── HSA Runtime types (hsa.h + hsa_ext_amd.h, ROCm 7.2.0) ──────────── */

typedef uint32_t hsa_status_t;
typedef struct { uint64_t handle; } hsa_agent_t;

#define HSA_STATUS_SUCCESS              0

/* hsa_agent_get_info attribute IDs (core) */
#define HSA_AGENT_INFO_WAVEFRONT_SIZE       6   /* uint32_t */
#define HSA_AGENT_INFO_WORKGROUP_MAX_SIZE   8   /* uint32_t */
#define HSA_AGENT_INFO_DEVICE               17  /* uint32_t, 1 = GPU */

/* hsa_agent_get_info attribute IDs (AMD vendor extensions) */
#define HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT   0xA002  /* uint32_t */
#define HSA_AMD_AGENT_INFO_DRIVER_NODE_ID       0xA004  /* uint32_t */
#define HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU     0xA00A  /* uint32_t */
#define HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES   0xA010  /* bool */
#define HSA_AMD_AGENT_INFO_NUM_SDMA_ENG         0xA10A  /* uint32_t */

typedef hsa_status_t (*hsa_init_t)(void);
typedef hsa_status_t (*hsa_shut_down_t)(void);
typedef hsa_status_t (*hsa_agent_get_info_t)(hsa_agent_t, uint32_t, void*);
typedef hsa_status_t (*hsa_iterate_agents_t)(
    hsa_status_t (*callback)(hsa_agent_t, void*), void*);

/* Per-GPU data extracted from HSA, keyed by KFD driver_node_id */
typedef struct {
    uint32_t driver_node_id;
    uint32_t wavefront_size;
    uint32_t workgroup_max_size;
    uint32_t max_waves_per_cu;
    uint32_t num_sdma_eng;
    char     cooperative;
} HsaGpuInfo;

#define HSA_MAX_GPUS 32

static void*                   hsa_dl          = NULL;
static hsa_agent_get_info_t    hsa_info_fn     = NULL;
static HsaGpuInfo              hsa_gpus[HSA_MAX_GPUS];
static int                     hsa_gpu_count   = 0;


static hsa_status_t hsa_agent_cb(hsa_agent_t agent, void* data) {
    (void)data;
    uint32_t dev_type = 0;
    if (hsa_info_fn(agent, HSA_AGENT_INFO_DEVICE, &dev_type) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (dev_type != 1)  /* not a GPU */
        return HSA_STATUS_SUCCESS;
    if (hsa_gpu_count >= HSA_MAX_GPUS)
        return HSA_STATUS_SUCCESS;

    HsaGpuInfo* g = &hsa_gpus[hsa_gpu_count];
    memset(g, 0, sizeof(*g));

    hsa_info_fn(agent, HSA_AMD_AGENT_INFO_DRIVER_NODE_ID,   &g->driver_node_id);
    hsa_info_fn(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE,       &g->wavefront_size);
    hsa_info_fn(agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE,   &g->workgroup_max_size);
    hsa_info_fn(agent, HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU, &g->max_waves_per_cu);
    hsa_info_fn(agent, HSA_AMD_AGENT_INFO_NUM_SDMA_ENG,     &g->num_sdma_eng);

    uint32_t coop = 0;
    if (hsa_info_fn(agent, HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES, &coop) == HSA_STATUS_SUCCESS)
        g->cooperative = (char)(coop != 0);

    hsa_gpu_count++;
    return HSA_STATUS_SUCCESS;
}


static void try_load_hsa() {
    if (hsa_dl) return;

    hsa_dl = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_LOCAL);
    if (!hsa_dl)
        hsa_dl = dlopen("/opt/rocm/lib/libhsa-runtime64.so", RTLD_NOW | RTLD_LOCAL);
    if (!hsa_dl) return;

    hsa_init_t init_fn = (hsa_init_t)dlsym(hsa_dl, "hsa_init");
    hsa_iterate_agents_t iter_fn = (hsa_iterate_agents_t)dlsym(hsa_dl, "hsa_iterate_agents");
    hsa_info_fn = (hsa_agent_get_info_t)dlsym(hsa_dl, "hsa_agent_get_info");

    if (!init_fn || !iter_fn || !hsa_info_fn) {
        dlclose(hsa_dl); hsa_dl = NULL; hsa_info_fn = NULL; return;
    }

    if (init_fn() != HSA_STATUS_SUCCESS) {
        dlclose(hsa_dl); hsa_dl = NULL; hsa_info_fn = NULL; return;
    }

    hsa_gpu_count = 0;
    iter_fn(hsa_agent_cb, NULL);
}


static const HsaGpuInfo* find_hsa_gpu(uint32_t node_id) {
    for (int i = 0; i < hsa_gpu_count; i++) {
        if (hsa_gpus[i].driver_node_id == node_id)
            return &hsa_gpus[i];
    }
    return NULL;
}


/* ── AMD SMI + HSA state ─────────────────────────────────────────────── */

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

    try_load_hsa();  /* optional, soft-fail */
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
    amdsmi_asic_info_t asic = {0};

    if (smi_asic_fn(handle, &asic) == AMDSMI_STATUS_SUCCESS) {
        strncpy(obj->_name_storage, asic.market_name, 255);
        obj->_name_storage[255] = '\0';
        obj->sms_count = (asic.num_of_compute_units != 0xFFFFFFFF)
            ? (int)asic.num_of_compute_units : 0;

        /* extract major/minor from target_graphics_version (hex gfx ID, e.g. 0x950) */
        uint64_t v = asic.target_graphics_version;
        if (v != 0 && v != 0xFFFFFFFFFFFFFFFFULL) {
            obj->major = (int)((v >> 8) & 0xFF);
            obj->minor = (int)((v >> 4) & 0xF);
        }
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
                    obj->l2_cache_size = (int)cache_info.cache[i].cache_size * 1024; /* KB → bytes */
                    break;
                }
            }
        }
    }

    strcpy(obj->_provider_storage, "HIP");
    obj->index = index;

    /* defaults for fields that may be populated by HSA below */
    obj->sm_threads = 0;
    obj->sm_shared_memory = 0;        /* not available without HIP runtime */
    obj->sm_registers = 0;            /* not available without HIP runtime */
    obj->sm_blocks = 0;               /* not available without HIP runtime */
    obj->block_threads = 0;
    obj->block_shared_memory = 0;     /* not available without HIP runtime */
    obj->block_registers = 0;         /* not available without HIP runtime */
    obj->warp_size = 64;              /* AMD wavefront size is always 64 */
    obj->concurrent_kernels = 1;      /* always true for GCN+ */
    obj->async_engines_count = 0;
    obj->cooperative = 0;

    /* backfill from HSA runtime if available, matched by KFD node ID */
    if (smi_kfd_fn) {
        amdsmi_kfd_info_t kfd = {0};
        if (smi_kfd_fn(handle, &kfd) == AMDSMI_STATUS_SUCCESS
            && kfd.node_id != 0xFFFFFFFF) {
            const HsaGpuInfo* hsa = find_hsa_gpu(kfd.node_id);
            if (hsa) {
                obj->warp_size = (int)hsa->wavefront_size;
                obj->block_threads = (int)hsa->workgroup_max_size;
                obj->sm_threads = (int)(hsa->max_waves_per_cu * hsa->wavefront_size);
                obj->async_engines_count = (int)hsa->num_sdma_eng;
                obj->cooperative = hsa->cooperative;
            }
        }
    }

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
    amdsmi_proc_info_t* infos = (amdsmi_proc_info_t*)calloc(num_procs, sizeof(amdsmi_proc_info_t));
    if (!infos) return -1;

    if (smi_proc_list_fn(gpu_handles[index], &num_procs, infos) != AMDSMI_STATUS_SUCCESS) {
        free(infos);
        return -1;
    }

    int n = 0;
    for (uint32_t i = 0; i < num_procs && n < max_count; i++) {
        pids[n++] = (int)infos[i].pid;
    }
    *count = n;
    free(infos);
    return 0;
}


int amdsmiGetGfxVersion(int index, char* gfx, int max_len) {
    if (try_load_amdsmi() != 0 || index < 0 || index >= gpu_count) return -1;

    amdsmi_asic_info_t asic = {0};
    if (smi_asic_fn(gpu_handles[index], &asic) != AMDSMI_STATUS_SUCCESS) return -1;

    if (asic.target_graphics_version == 0xFFFFFFFFFFFFFFFFULL) return -1;

    snprintf(gfx, max_len, "%x", (unsigned)asic.target_graphics_version);
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

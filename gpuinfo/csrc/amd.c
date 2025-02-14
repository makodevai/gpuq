
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>

#include "types.h"


#define hipGetDeviceProperties hipGetDevicePropertiesR0600
#define hipDeviceProp_t hipDeviceProp_tR0600
#define hipChooseDevice hipChooseDeviceR0600


typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;     ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1;  ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics : 1;     ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1;  ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd : 1;  ///< 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1;  ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics : 1;  ///< 64-bit integer atomics for shared memory.

    // Doubles
    unsigned hasDoubles : 1;  ///< Double-precision floating point.

    // Warp cross-lane operations
    unsigned hasWarpVote : 1;     ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot : 1;   ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle : 1;  ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift : 1;  ///< Funnel two words into one with shift&mask caps.

    // Sync
    unsigned hasThreadFenceSystem : 1;  ///< __threadfence_system.
    unsigned hasSyncThreadsExt : 1;     ///< __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    unsigned hasSurfaceFuncs : 1;        ///< Surface functions.
    unsigned has3dGrid : 1;              ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism : 1;  ///< Dynamic parallelism.
} hipDeviceArch_t;


typedef struct hipUUID_t {
    char bytes[16];
} hipUUID;


typedef struct hipDeviceProp_t {
    char name[256];                   ///< Device name.
    hipUUID uuid;                     ///< UUID of a device
    char luid[8];                     ///< 8-byte unique identifier. Only valid on windows
    unsigned int luidDeviceNodeMask;  ///< LUID node mask
    size_t totalGlobalMem;            ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;         ///< Size of shared memory per block (in bytes).
    int regsPerBlock;                 ///< Registers per block.
    int warpSize;                     ///< Warp size.
    size_t memPitch;                  ///< Maximum pitch in bytes allowed by memory copies
    int maxThreadsPerBlock;           ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];             ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];               ///< Max grid dimensions (XYZ).
    int clockRate;                    ///< Max clock frequency of the multiProcessors in khz.
    size_t totalConstMem;             ///< Size of shared constant memory region on the device
    int major;  ///< Major compute capability.  On HCC, this is an approximation and features may
    int minor;  ///< Minor compute capability.  On HCC, this is an approximation and features may
    size_t textureAlignment;       ///< Alignment requirement for textures
    size_t texturePitchAlignment;  ///< Pitch alignment requirement for texture references bound to
    int deviceOverlap;             ///< Deprecated. Use asyncEngineCount instead
    int multiProcessorCount;       ///< Number of multi-processors (compute units).
    int kernelExecTimeoutEnabled;  ///< Run time limit for kernels executed on the device
    int integrated;                ///< APU vs dGPU
    int canMapHostMemory;          ///< Check whether HIP can map host memory
    int computeMode;               ///< Compute mode.
    int maxTexture1D;              ///< Maximum number of elements in 1D images
    int maxTexture1DMipmap;        ///< Maximum 1D mipmap texture size
    int maxTexture1DLinear;        ///< Maximum size for 1D textures bound to linear memory
    int maxTexture2D[2];  ///< Maximum dimensions (width, height) of 2D images, in image elements
    int maxTexture2DMipmap[2];  ///< Maximum number of elements in 2D array mipmap of images
    int maxTexture2DLinear[3];  ///< Maximum 2D tex dimensions if tex are bound to pitched memory
    int maxTexture2DGather[2];  ///< Maximum 2D tex dimensions if gather has to be performed
    int maxTexture3D[3];  ///< Maximum dimensions (width, height, depth) of 3D images, in image
    int maxTexture3DAlt[3];           ///< Maximum alternate 3D texture dims
    int maxTextureCubemap;            ///< Maximum cubemap texture dims
    int maxTexture1DLayered[2];       ///< Maximum number of elements in 1D array images
    int maxTexture2DLayered[3];       ///< Maximum number of elements in 2D array images
    int maxTextureCubemapLayered[2];  ///< Maximum cubemaps layered texture dims
    int maxSurface1D;                 ///< Maximum 1D surface size
    int maxSurface2D[2];              ///< Maximum 2D surface size
    int maxSurface3D[3];              ///< Maximum 3D surface size
    int maxSurface1DLayered[2];       ///< Maximum 1D layered surface size
    int maxSurface2DLayered[3];       ///< Maximum 2D layared surface size
    int maxSurfaceCubemap;            ///< Maximum cubemap surface size
    int maxSurfaceCubemapLayered[2];  ///< Maximum cubemap layered surface size
    size_t surfaceAlignment;          ///< Alignment requirement for surface
    int concurrentKernels;         ///< Device can possibly execute multiple kernels concurrently.
    int ECCEnabled;                ///< Device has ECC support enabled
    int pciBusID;                  ///< PCI Bus ID.
    int pciDeviceID;               ///< PCI Device ID.
    int pciDomainID;               ///< PCI Domain ID
    int tccDriver;                 ///< 1:If device is Tesla device using TCC driver, else 0
    int asyncEngineCount;          ///< Number of async engines
    int unifiedAddressing;         ///< Does device and host share unified address space
    int memoryClockRate;           ///< Max global memory clock frequency in khz.
    int memoryBusWidth;            ///< Global memory bus width in bits.
    int l2CacheSize;               ///< L2 cache size.
    int persistingL2CacheMaxSize;  ///< Device's max L2 persisting lines in bytes
    int maxThreadsPerMultiProcessor;    ///< Maximum resident threads per multi-processor.
    int streamPrioritiesSupported;      ///< Device supports stream priority
    int globalL1CacheSupported;         ///< Indicates globals are cached in L1
    int localL1CacheSupported;          ///< Locals are cahced in L1
    size_t sharedMemPerMultiprocessor;  ///< Amount of shared memory available per multiprocessor.
    int regsPerMultiprocessor;          ///< registers available per multiprocessor
    int managedMemory;         ///< Device supports allocating managed memory on this system
    int isMultiGpuBoard;       ///< 1 if device is on a multi-GPU board, 0 if not.
    int multiGpuBoardGroupID;  ///< Unique identifier for a group of devices on same multiboard GPU
    int hostNativeAtomicSupported;         ///< Link between host and device supports native atomics
    int singleToDoublePrecisionPerfRatio;  ///< Deprecated. CUDA only.
    int pageableMemoryAccess;              ///< Device supports coherently accessing pageable memory
    int concurrentManagedAccess;  ///< Device can coherently access managed memory concurrently with
    int computePreemptionSupported;         ///< Is compute preemption supported on the device
    int canUseHostPointerForRegisteredMem;  ///< Device can access host registered memory with same
    int cooperativeLaunch;                  ///< HIP device supports cooperative launch
    int cooperativeMultiDeviceLaunch;       ///< HIP device supports cooperative launch on multiple
    size_t sharedMemPerBlockOptin;  ///< Per device m ax shared mem per block usable by special opt in
    int pageableMemoryAccessUsesHostPageTables;  ///< Device accesses pageable memory via the host's
    int directManagedMemAccessFromHost;  ///< Host can directly access managed memory on the device
    int maxBlocksPerMultiProcessor;      ///< Max number of blocks on CU
    int accessPolicyMaxWindowSize;       ///< Max value of access policy window
    size_t reservedSharedMemPerBlock;    ///< Shared memory reserved by driver per block
    int hostRegisterSupported;           ///< Device supports hipHostRegister
    int sparseHipArraySupported;         ///< Indicates if device supports sparse hip arrays
    int hostRegisterReadOnlySupported;   ///< Device supports using the hipHostRegisterReadOnly flag
    int timelineSemaphoreInteropSupported;  ///< Indicates external timeline semaphore support
    int memoryPoolsSupported;  ///< Indicates if device supports hipMallocAsync and hipMemPool APIs
    int gpuDirectRDMASupported;                    ///< Indicates device support of RDMA APIs
    unsigned int gpuDirectRDMAFlushWritesOptions;  ///< Bitmask to be interpreted according to
    int gpuDirectRDMAWritesOrdering;               ///< value of hipGPUDirectRDMAWritesOrdering
    unsigned int memoryPoolSupportedHandleTypes;  ///< Bitmask of handle types support with mempool based IPC
    int deferredMappingHipArraySupported;  ///< Device supports deferred mapping HIP arrays and HIP
    int ipcEventSupported;                 ///< Device supports IPC events
    int clusterLaunch;                     ///< Device supports cluster launch
    int unifiedFunctionPointers;           ///< Indicates device supports unified function pointers
    int reserved[63];                      ///< CUDA Reserved.

    int hipReserved[32];  ///< Reserved for adding new entries for HIP/CUDA.

    /* HIP Only struct members */
    char gcnArchName[256];                    ///< AMD GCN Arch Name. HIP Only.
    size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per CU. HIP Only.
    int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                               ///< instructions.  New for HIP.
    hipDeviceArch_t arch;      ///< Architectural feature flags.  New for HIP.
    unsigned int* hdpMemFlushCntl;            ///< Addres of HDP_MEM_COHERENCY_FLUSH_CNTL register
    unsigned int* hdpRegFlushCntl;            ///< Addres of HDP_REG_COHERENCY_FLUSH_CNTL register
    int cooperativeMultiDeviceUnmatchedFunc;  ///< HIP device supports cooperative launch on
                                              ///< multiple
                                              /// devices with unmatched functions
    int cooperativeMultiDeviceUnmatchedGridDim;    ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched grid dimensions
    int cooperativeMultiDeviceUnmatchedBlockDim;   ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched block dimensions
    int cooperativeMultiDeviceUnmatchedSharedMem;  ///< HIP device supports cooperative launch on
                                                   ///< multiple
                                                   /// devices with unmatched shared memories
    int isLargeBar;                                ///< 1: if it is a large PCI bar device, else 0
    int asicRevision;                              ///< Revision of the GPU in this device
} hipDeviceProp_t;


typedef enum hipError_t {
    hipSuccess = 0,  ///< Successful completion.
    hipErrorInvalidValue = 1,  ///< One or more of the parameters passed to the API call is NULL
                               ///< or not in an acceptable range.
    hipErrorOutOfMemory = 2,   ///< out of memory range.
    // Deprecated
    hipErrorMemoryAllocation = 2,  ///< Memory allocation error.
    hipErrorNotInitialized = 3,    ///< Invalid not initialized
    // Deprecated
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,      ///< Deinitialized
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,  ///< Invalide configuration
    hipErrorInvalidPitchValue = 12,   ///< Invalid pitch value
    hipErrorInvalidSymbol = 13,   ///< Invalid symbol
    hipErrorInvalidDevicePointer = 17,  ///< Invalid Device Pointer
    hipErrorInvalidMemcpyDirection = 21,  ///< Invalid memory copy direction
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,  ///< Invalid device function
    hipErrorNoDevice = 100,  ///< Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice = 101,  ///< DeviceID must be in range from 0 to compute-devices.
    hipErrorInvalidImage = 200,   ///< Invalid image
    hipErrorInvalidContext = 201,  ///< Produced when input context is invalid.
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    // Deprecated
    hipErrorMapBufferObjectFailed = 205,  ///< Produced when the IPC memory attach failed from ROCr.
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,   ///< Unsupported limit
    hipErrorContextAlreadyInUse = 216,   ///< The context is already in use
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,  ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,   ///< Invalid source.
    hipErrorFileNotFound = 301,   ///< the file is not found.
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,   ///< Failed to initialize shared object.
    hipErrorOperatingSystem = 304,   ///< Not the correct operating system
    hipErrorInvalidHandle = 400,  ///< Invalide handle
    // Deprecated
    hipErrorInvalidResourceHandle = 400,  ///< Resource handle (hipEvent_t or hipStream_t) invalid.
    hipErrorIllegalState = 401, ///< Resource required is not in a valid state to perform operation.
    hipErrorNotFound = 500,   ///< Not found
    hipErrorNotReady = 600,  ///< Indicates that asynchronous operations enqueued earlier are not
                             ///< ready.  This is not actually an error, but is used to distinguish
                             ///< from hipSuccess (which indicates completion).  APIs that return
                             ///< this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,  ///< Out of resources error.
    hipErrorLaunchTimeOut = 702,   ///< Timeout for the launch.
    hipErrorPeerAccessAlreadyEnabled = 704,  ///< Peer access was already enabled from the current
                                             ///< device.
    hipErrorPeerAccessNotEnabled = 705,  ///< Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess = 708,   ///< The process is active.
    hipErrorContextIsDestroyed = 709,   ///< The context is already destroyed
    hipErrorAssert = 710,  ///< Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered = 712,  ///< Produced when trying to lock a page-locked
                                                ///< memory.
    hipErrorHostMemoryNotRegistered = 713,  ///< Produced when trying to unlock a non-page-locked
                                            ///< memory.
    hipErrorLaunchFailure = 719,  ///< An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge = 720,  ///< This error indicates that the number of blocks
                                              ///< launched per grid for a kernel that was launched
                                              ///< via cooperative launch APIs exceeds the maximum
                                              ///< number of allowed blocks for the current device.
    hipErrorNotSupported = 801,  ///< Produced when the hip API is not supported/implemented
    hipErrorStreamCaptureUnsupported = 900,  ///< The operation is not permitted when the stream
                                             ///< is capturing.
    hipErrorStreamCaptureInvalidated = 901,  ///< The current capture sequence on the stream
                                             ///< has been invalidated due to a previous error.
    hipErrorStreamCaptureMerge = 902,  ///< The operation would have resulted in a merge of
                                       ///< two independent capture sequences.
    hipErrorStreamCaptureUnmatched = 903,  ///< The capture was not initiated in this stream.
    hipErrorStreamCaptureUnjoined = 904,  ///< The capture sequence contains a fork that was not
                                          ///< joined to the primary stream.
    hipErrorStreamCaptureIsolation = 905,  ///< A dependency would have been created which crosses
                                           ///< the capture sequence boundary. Only implicit
                                           ///< in-stream ordering dependencies  are allowed
                                           ///< to cross the boundary
    hipErrorStreamCaptureImplicit = 906,  ///< The operation would have resulted in a disallowed
                                          ///< implicit dependency on a current capture sequence
                                          ///< from hipStreamLegacy.
    hipErrorCapturedEvent = 907,  ///< The operation is not permitted on an event which was last
                                  ///< recorded in a capturing stream.
    hipErrorStreamCaptureWrongThread = 908,  ///< A stream capture sequence not initiated with
                                             ///< the hipStreamCaptureModeRelaxed argument to
                                             ///< hipStreamBeginCapture was passed to
                                             ///< hipStreamEndCapture in a different thread.
    hipErrorGraphExecUpdateFailure = 910,  ///< This error indicates that the graph update
                                           ///< not performed because it included changes which
                                           ///< violated constraintsspecific to instantiated graph
                                           ///< update.
    hipErrorInvalidChannelDescriptor = 911,  ///< Invalid channel descriptor.
    hipErrorInvalidTexture = 912,  ///< Invalid texture.
    hipErrorUnknown = 999,  ///< Unknown error.
    // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory = 1052,  ///< HSA runtime memory call returned error.  Typically not seen
                                   ///< in production systems.
    hipErrorRuntimeOther = 1053,  ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
    hipErrorTbd  ///< Marker that more error codes are needed.
} hipError_t;


typedef hipError_t (*hipGetDeviceProperties_t)(hipDeviceProp_t* prop, int deviceId);
typedef hipError_t (*hipGetDeviceCount_t)(int* count);


static void* hip_runtime_dl = NULL;
static hipGetDeviceCount_t device_count_fn = NULL;
static hipGetDeviceProperties_t device_props_fn = NULL;


static int try_load_hipruntime() {
    if (!hip_runtime_dl) {
        hip_runtime_dl = dlopen("libamdhip64.so", RTLD_NOW|RTLD_LOCAL);
        if (!hip_runtime_dl)
            return -1;
    }

    if (!device_count_fn) {
        device_count_fn = (hipGetDeviceCount_t)dlsym(hip_runtime_dl, "hipGetDeviceCount");
        if (!device_count_fn)
            return -2;
    }

    if (!device_props_fn) {
        device_props_fn = (hipGetDeviceProperties_t)dlsym(hip_runtime_dl, "hipGetDevicePropertiesR0600");
        if (!device_props_fn)
            return -3;
    }

    return 0;
}


int amdPresent() {
    int status = try_load_hipruntime();
    if (status)
        return 0;
    return 1;
}


int amdGetDeviceCount(int* count) {
    int status = try_load_hipruntime();
    if (status)
        return status;

    hipError_t e = device_count_fn(count);
    if (e != hipSuccess)
        return (int)e;

    return 0;
}


int amdGetDeviceProps(int index, GpuProp* obj) {
    int status = try_load_hipruntime();
    if (status)
        return status;

    hipDeviceProp_t deviceProp;
    hipError_t e = device_props_fn(&deviceProp, index);
    if (e != hipSuccess) {
        return (int)e;
    }

    strcpy(obj->_provider_storage, "HIP");
    obj->index = index;
    memcpy(obj->_name_storage, deviceProp.name, 256);
    obj->major = deviceProp.major;
    obj->minor = deviceProp.minor;
    obj->total_memory = deviceProp.totalGlobalMem;
    obj->sms_count = deviceProp.multiProcessorCount;
    obj->sm_threads = deviceProp.maxThreadsPerMultiProcessor;
    // obj->sm_shared_memory = deviceProp.sharedMemPerMultiprocessor;
    obj->sm_shared_memory = deviceProp.maxSharedMemoryPerMultiProcessor; // hip specific, seems to make more sense
    obj->sm_registers = deviceProp.regsPerMultiprocessor;
    obj->sm_blocks = deviceProp.maxBlocksPerMultiProcessor;
    obj->block_threads = deviceProp.maxThreadsPerBlock;
    obj->block_shared_memory = deviceProp.sharedMemPerBlock;
    obj->block_registers = deviceProp.regsPerBlock;
    obj->warp_size = deviceProp.warpSize;
    obj->l2_cache_size = deviceProp.l2CacheSize;
    obj->concurrent_kernels = (char)deviceProp.concurrentKernels;
    obj->async_engines_count = deviceProp.asyncEngineCount;
    obj->cooperative = (char)deviceProp.cooperativeLaunch;

    return 0;
}


void amdCleanup() {
    if (hip_runtime_dl) {
        dlclose(hip_runtime_dl);
    }

    hip_runtime_dl = NULL;
    device_count_fn = NULL;
    device_props_fn = NULL;
}

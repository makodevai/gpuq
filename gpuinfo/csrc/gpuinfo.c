#include "types.h"
#include <stddef.h>

#include "patchlevel.h"

// for Python <3.12
#if PY_VERSION_HEX < 0x030c0000
#include "structmember.h"
#define Py_T_UINT T_UINT
#define Py_T_ULONG T_ULONG
#define Py_T_ULONGLONG T_ULONGULONG
#define Py_T_STRING T_STRING
#define Py_T_INT T_INT
#define Py_T_BOOL T_BOOL
#endif

#if SIZE_MAX == UINT_MAX
  #define Py_T_SIZET Py_T_UINT
#elif SIZE_MAX == ULONG_MAX
  #define Py_T_SIZET Py_T_ULONG
#elif SIZE_MAX == ULLONG_MAX
  #define Py_T_SIZET Py_T_ULONGLONG
#else
  #error "Could not determine size_t size!"
#endif





static PyMemberDef GpuPropMembers[] = {
    {"ord", Py_T_INT, offsetof(GpuProp, ord), 0, "GPU ordinal, across all devices and providers, specific to this package"},
    {"provider", Py_T_STRING, offsetof(GpuProp, provider), 0, "GPU provider (cuda, hip, etc.)"},
    {"index", Py_T_INT, offsetof(GpuProp, index), 0, "GPU index for its provider, subject to *_VISIBLE_DEVICES"},
    {"name", Py_T_STRING, offsetof(GpuProp, name), 0, "GPU model name"},
    {"major", Py_T_INT, offsetof(GpuProp, major), 0, "Model major number"},
    {"minor", Py_T_INT, offsetof(GpuProp, minor), 0, "Model minor number"},
    {"total_memory", Py_T_SIZET, offsetof(GpuProp, total_memory), 0, "Total global memory (in bytes)"},
    {"sms_count", Py_T_INT, offsetof(GpuProp, sms_count), 0, "Number of multiprocessors"},
    {"sm_threads", Py_T_INT, offsetof(GpuProp, sm_threads), 0, "Number of threads per multiprocessor"},
    {"sm_shared_memory", Py_T_SIZET, offsetof(GpuProp, sm_shared_memory), 0, "Shared memory per multiprocessor (in bytes)"},
    {"sm_registers", Py_T_INT, offsetof(GpuProp, sm_registers), 0, "Number of registers per multiprocessor"},
    {"sm_blocks", Py_T_INT, offsetof(GpuProp, sm_blocks), 0, "Maximum number of blocks per multiprocessor"},
    {"block_threads", Py_T_INT, offsetof(GpuProp, block_threads), 0, "Maximum number of threads per block"},
    {"block_shared_memory", Py_T_SIZET, offsetof(GpuProp, block_shared_memory), 0, "Shared memory per block (in bytes)"},
    {"block_registers", Py_T_INT, offsetof(GpuProp, block_registers), 0, "Number of registers per block"},
    {"warp_size", Py_T_INT, offsetof(GpuProp, warp_size), 0, "Warp size"},
    {"l2_cache_size", Py_T_INT, offsetof(GpuProp, l2_cache_size), 0, "L2 cache size"},
    {"concurrent_kernels", Py_T_BOOL, offsetof(GpuProp, concurrent_kernels), 0, "Whether the device supports concurrent kernels"},
    {"async_engines_count", Py_T_INT, offsetof(GpuProp, async_engines_count), 0, "Number of asynchronous engines"},
    {"cooperative", Py_T_BOOL, offsetof(GpuProp, cooperative), 0, "Whether the device supports cooperative launches"},
    {NULL}
};


static PyTypeObject GpuPropType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gpuinfo.C.Properties",
    .tp_doc = PyDoc_STR("A structure holding device properties of a GPU."),
    .tp_basicsize = sizeof(GpuProp),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_members = GpuPropMembers,
};


static int cudaDevices = 0;
static int amdDevices = 0;


static int get_gpu_count() {
    int status = cudaGetDeviceCount(&cudaDevices);
    if (status < 0)
        cudaDevices = 0;
    else if (status > 0) {
        PyErr_SetString(PyExc_RuntimeError, "cudaGetDeviceCount failed");
        return -1;
    }

    status = amdGetDeviceCount(&amdDevices);
    if (status < 0)
        amdDevices = 0;
    else if (status > 0) {
        PyErr_SetString(PyExc_RuntimeError, "hipGetDeviceCount failed");
        return -1;
    }

    return cudaDevices + amdDevices;
}


static PyObject*
gpuinfo_checkcuda(PyObject* self, PyObject* args) {
    return PyLong_FromLong(checkCuda());
}


static PyObject*
gpuinfo_checkamd(PyObject* self, PyObject* args) {
    return PyLong_FromLong(checkAmd());
}



static PyObject*
gpuinfo_count(PyObject* self, PyObject* args) {
    int count = get_gpu_count();
    if (count < 0)
        return NULL;
    return PyLong_FromLong(count);
}


static PyObject*
gpuinfo_get(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "gpuinfo.C.get takes exactly 1 positional argument only.");
        return NULL;
    }

    int gpu_id = -1;
    if (!PyArg_Parse(args[0], "i:gpuinfo.C.get", &gpu_id))
        return NULL;

    int dev_count = get_gpu_count();
    if (dev_count < 0)
        return NULL;

    if (!dev_count) {
        PyErr_SetString(PyExc_RuntimeError, "No GPUs available");
        return NULL;
    }

    if (gpu_id < 0 || gpu_id >= dev_count) {
        PyErr_SetString(PyExc_ValueError, "Invalid GPU index");
        return NULL;
    }

    GpuProp* obj = (GpuProp*)PyObject_CallNoArgs((PyObject*)&GpuPropType);
    if (obj == NULL)
        return NULL;

    obj->ord = gpu_id;
    obj->name = &obj->_name_storage[0];
    obj->provider = &obj->_provider_storage[0];

    int status = 0;
    if (gpu_id < cudaDevices) {
        status = cudaGetDeviceProps(gpu_id, obj);
    } else { // gpu_id >= cudaDevice && gpu_id < cudaDevice+amdDevices
        status = amdGetDeviceProps(gpu_id-cudaDevices, obj);
    }

    if (status) {
        PyErr_SetString(PyExc_RuntimeError, "Could not query device properties.");
        return NULL;
    }

    return (PyObject*)obj;
}


static PyMethodDef gpuinfo_methods[] = {
    {"checkcuda", gpuinfo_checkcuda, METH_NOARGS, "Return status code for CUDA runtime."},
    {"checkamd", gpuinfo_checkamd, METH_NOARGS, "Return status code for HIP runtime."},
    {"count", gpuinfo_count, METH_NOARGS, "Return the number of GPUs."},
    {"get", (PyCFunction)gpuinfo_get, METH_FASTCALL, "Return properties of a GPU with a given index."},
    {NULL, NULL, 0, NULL}
};


static PyModuleDef gpuinfo = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "gpuinfo.C",
    .m_doc = "Module to query information about available gpus.",
    .m_size = -1,
    .m_methods = gpuinfo_methods,
};


PyMODINIT_FUNC
PyInit_C(void)
{
    PyObject *m;
    if (PyType_Ready(&GpuPropType) < 0)
        return NULL;

    m = PyModule_Create(&gpuinfo);
    if (m == NULL)
        return NULL;

    if (PyModule_AddObjectRef(m, "Properties", (PyObject*)&GpuPropType) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

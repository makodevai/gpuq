#include "types.h"
#include <stddef.h>
#include <dlfcn.h>

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
    {"uuid", Py_T_STRING, offsetof(GpuProp, uuid), 0, "Device UUID"},
    {"provider", Py_T_STRING, offsetof(GpuProp, provider), 0, "GPU provider (cuda, hip, etc.)"},
    {"index", Py_T_INT, offsetof(GpuProp, index), 0, "GPU index for its provider, subject to *_VISIBLE_DEVICES"},
    {"name", Py_T_STRING, offsetof(GpuProp, name), 0, "GPU model name"},
    {"major", Py_T_INT, offsetof(GpuProp, major), 0, "Model major number"},
    {"minor", Py_T_INT, offsetof(GpuProp, minor), 0, "Model minor number"},
    {"total_memory", Py_T_SIZET, offsetof(GpuProp, total_memory), 0, "Total global memory (in bytes)"},
    {"sms_count", Py_T_INT, offsetof(GpuProp, sms_count), 0, "Number of multiprocessors / compute units"},
    {"l2_cache_size", Py_T_INT, offsetof(GpuProp, l2_cache_size), 0, "L2 cache size (in KB)"},
    {NULL}
};


static PyTypeObject GpuPropType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gpuq.C.Properties",
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
    if (status != 0)
        cudaDevices = 0;

    status = amdGetDeviceCount(&amdDevices);
    if (status != 0)
        amdDevices = 0;

    return cudaDevices + amdDevices;
}


static PyObject*
gpuq_checkcuda(PyObject* self, PyObject* args) {
    int status = checkCuda();
    if (status) {
        const char* error_str = cudaGetDlError();
        return PyUnicode_FromFormat("%s:\n%s", "Could not load libnvidia-ml.so", (error_str ? error_str : "(unknown)"));
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) || count <= 0)
        return PyUnicode_InternFromString("No CUDA-capable devices detected (via NVML)");
    return PyUnicode_InternFromString("");
}


static PyObject*
gpuq_checkamd(PyObject* self, PyObject* args) {
    int status = checkAmd();
    if (status) {
        const char* error_str = amdGetDlError();
        return PyUnicode_FromFormat("%s:\n%s", "Could not load libamd_smi.so", (error_str ? error_str : "(unknown)"));
    }
    int count = 0;
    if (amdGetDeviceCount(&count) || count <= 0)
        return PyUnicode_InternFromString("No AMD GPU devices detected (via AMD SMI)");
    return PyUnicode_InternFromString("");
}


static PyObject*
gpuq_count(PyObject* self, PyObject* args) {
    int count = get_gpu_count();
    return PyLong_FromLong(count);
}


static PyObject*
gpuq_cuda_count(PyObject* self, PyObject* args) {
    get_gpu_count();  /* ensure cudaDevices is populated */
    return PyLong_FromLong(cudaDevices);
}


static PyObject*
gpuq_amd_count(PyObject* self, PyObject* args) {
    get_gpu_count();  /* ensure amdDevices is populated */
    return PyLong_FromLong(amdDevices);
}


static PyObject*
gpuq_get(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "gpuq.C.get takes exactly 1 positional argument only.");
        return NULL;
    }

    int gpu_id = -1;
    if (!PyArg_Parse(args[0], "i:gpuq.C.get", &gpu_id))
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
    obj->uuid = &obj->_uuid_storage[0];
    obj->name = &obj->_name_storage[0];
    obj->provider = &obj->_provider_storage[0];

    int status = 0;
    if (gpu_id < cudaDevices) {
        status = cudaGetDeviceProps(gpu_id, obj);
    } else {
        status = amdGetDeviceProps(gpu_id - cudaDevices, obj);
    }

    if (status) {
        PyErr_SetString(PyExc_RuntimeError, "Could not query device properties.");
        return NULL;
    }

    return (PyObject*)obj;
}


static PyObject*
gpuq_nvml_utilisation(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "nvml_utilisation takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    int util = 0;
    if (nvmlGetRuntimeUtilisation(index, &util) != 0)
        return PyLong_FromLong(-1);
    return PyLong_FromLong(util);
}


static PyObject*
gpuq_nvml_used_memory(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "nvml_used_memory takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    unsigned long long used = 0;
    if (nvmlGetRuntimeMemory(index, &used) != 0)
        return PyLong_FromLong(-1);
    /* return in MiB to match previous nvidia-smi convention */
    return PyLong_FromUnsignedLongLong(used / (1024 * 1024));
}


static PyObject*
gpuq_nvml_pids(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "nvml_pids takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    int pids[128];
    int count = 0;
    nvmlGetRuntimePids(index, pids, &count, 128);

    PyObject* list = PyList_New(count);
    if (!list) return NULL;
    for (int i = 0; i < count; i++) {
        PyList_SET_ITEM(list, i, PyLong_FromLong(pids[i]));
    }
    return list;
}


static PyObject*
gpuq_amdsmi_utilisation(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "amdsmi_utilisation takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    int util = 0;
    if (amdsmiGetRuntimeUtilisation(index, &util))
        return PyLong_FromLong(-1);
    return PyLong_FromLong(util);
}


static PyObject*
gpuq_amdsmi_used_memory(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "amdsmi_used_memory takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    unsigned long long used = 0;
    if (amdsmiGetRuntimeMemory(index, &used))
        return PyLong_FromLong(-1);
    return PyLong_FromUnsignedLongLong(used / (1024 * 1024));
}


static PyObject*
gpuq_amdsmi_pids(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "amdsmi_pids takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    int pids[128];
    int count = 0;
    amdsmiGetRuntimePids(index, pids, &count, 128);

    PyObject* list = PyList_New(count);
    if (!list) return NULL;
    for (int i = 0; i < count; i++) {
        PyList_SET_ITEM(list, i, PyLong_FromLong(pids[i]));
    }
    return list;
}


static PyObject*
gpuq_amdsmi_gfx(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "amdsmi_gfx takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    char gfx[64] = {0};
    if (amdsmiGetGfxVersion(index, gfx, sizeof(gfx)))
        return PyUnicode_FromString("");
    return PyUnicode_FromString(gfx);
}


static PyObject*
gpuq_amdsmi_drm(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "amdsmi_drm takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    int drm = -1;
    amdsmiGetDrmRender(index, &drm);
    return PyLong_FromLong(drm);
}


static PyObject*
gpuq_amdsmi_node_id(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "amdsmi_node_id takes exactly 1 argument");
        return NULL;
    }
    int index = -1;
    if (!PyArg_Parse(args[0], "i", &index)) return NULL;

    int node_id = -1;
    amdsmiGetNodeId(index, &node_id);
    return PyLong_FromLong(node_id);
}


static PyMethodDef gpuq_methods[] = {
    {"checkcuda", gpuq_checkcuda, METH_NOARGS, "Return status code for NVML (NVIDIA)."},
    {"checkamd", gpuq_checkamd, METH_NOARGS, "Return status code for AMD SMI."},
    {"count", gpuq_count, METH_NOARGS, "Return the number of GPUs."},
    {"_cuda_count", gpuq_cuda_count, METH_NOARGS, "(internal) CUDA device count via NVML."},
    {"_amd_count", gpuq_amd_count, METH_NOARGS, "(internal) AMD device count via AMD SMI."},
    {"get", (PyCFunction)gpuq_get, METH_FASTCALL, "Return properties of a GPU with a given index."},
    {"_nvml_utilisation", (PyCFunction)gpuq_nvml_utilisation, METH_FASTCALL, "(internal) GPU utilisation % for NVIDIA device at index."},
    {"_nvml_used_memory", (PyCFunction)gpuq_nvml_used_memory, METH_FASTCALL, "(internal) used memory in MiB for NVIDIA device at index."},
    {"_nvml_pids", (PyCFunction)gpuq_nvml_pids, METH_FASTCALL, "(internal) list of PIDs using NVIDIA device at index."},
    {"_amdsmi_utilisation", (PyCFunction)gpuq_amdsmi_utilisation, METH_FASTCALL, "(internal) GPU utilisation % for AMD device at index."},
    {"_amdsmi_used_memory", (PyCFunction)gpuq_amdsmi_used_memory, METH_FASTCALL, "(internal) used memory in MiB for AMD device at index."},
    {"_amdsmi_pids", (PyCFunction)gpuq_amdsmi_pids, METH_FASTCALL, "(internal) list of PIDs using AMD device at index."},
    {"_amdsmi_gfx", (PyCFunction)gpuq_amdsmi_gfx, METH_FASTCALL, "(internal) GFX version string for AMD device at index."},
    {"_amdsmi_drm", (PyCFunction)gpuq_amdsmi_drm, METH_FASTCALL, "(internal) DRM render minor for AMD device at index."},
    {"_amdsmi_node_id", (PyCFunction)gpuq_amdsmi_node_id, METH_FASTCALL, "(internal) KFD node ID for AMD device at index."},
    {NULL, NULL, 0, NULL}
};


static PyModuleDef gpuq = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "gpuq.C",
    .m_doc = "Module to query information about available gpus.",
    .m_size = -1,
    .m_methods = gpuq_methods,
};


PyMODINIT_FUNC
PyInit_C(void)
{
    PyObject *m;
    if (PyType_Ready(&GpuPropType) < 0)
        return NULL;

    m = PyModule_Create(&gpuq);
    if (m == NULL)
        return NULL;

    if (PyModule_AddObjectRef(m, "Properties", (PyObject*)&GpuPropType) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

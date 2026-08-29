/*
 * Process-local NVML compatibility shim for ren5.
 *
 * ren5's third RTX 3090 has fallen off the PCIe bus. NVML still reports three
 * devices, and NCCL aborts while opening the broken final device even when
 * CUDA_VISIBLE_DEVICES only exposes GPU 0 and GPU 1. This proxy forwards NVML
 * to the real driver while clamping the reported device count. It is loaded
 * only by the EndoWAM training process through LD_LIBRARY_PATH; it does not
 * alter the driver or system-wide NVML installation.
 */

#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef REAL_NVML_PATH
#error "REAL_NVML_PATH must point to the system libnvidia-ml.so.1"
#endif

static pthread_once_t library_once = PTHREAD_ONCE_INIT;
static void *real_library = NULL;

static void open_real_library(void) {
    real_library = dlopen(REAL_NVML_PATH, RTLD_NOW | RTLD_LOCAL);
    if (real_library == NULL) {
        fprintf(stderr, "ren5 NVML filter: cannot open %s: %s\n", REAL_NVML_PATH, dlerror());
    }
}

static void *real_symbol(const char *name) {
    pthread_once(&library_once, open_real_library);
    return real_library == NULL ? NULL : dlsym(real_library, name);
}

static unsigned int visible_device_limit(void) {
    const char *value = getenv("REN5_NVML_MAX_DEVICES");
    if (value == NULL || *value == '\0') return 2;
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 || parsed > 64) return 2;
    return (unsigned int)parsed;
}

#define FORWARD_NVML(name, args, call_args)                    \
    nvmlReturn_t name args {                                  \
        typedef nvmlReturn_t (*function_t) args;               \
        function_t function = (function_t)real_symbol(#name);  \
        if (function == NULL) return NVML_ERROR_UNKNOWN;       \
        return function call_args;                             \
    }

nvmlReturn_t nvmlInit(void) {
    typedef nvmlReturn_t (*function_t)(void);
    function_t function = (function_t)real_symbol("nvmlInit");
    return function == NULL ? NVML_ERROR_UNKNOWN : function();
}

nvmlReturn_t nvmlInit_v2(void) {
    typedef nvmlReturn_t (*function_t)(void);
    function_t function = (function_t)real_symbol("nvmlInit_v2");
    return function == NULL ? NVML_ERROR_UNKNOWN : function();
}

nvmlReturn_t nvmlShutdown(void) {
    typedef nvmlReturn_t (*function_t)(void);
    function_t function = (function_t)real_symbol("nvmlShutdown");
    return function == NULL ? NVML_ERROR_UNKNOWN : function();
}

static nvmlReturn_t filtered_count(const char *symbol_name, unsigned int *device_count) {
    typedef nvmlReturn_t (*function_t)(unsigned int *);
    function_t function = (function_t)real_symbol(symbol_name);
    if (function == NULL) return NVML_ERROR_UNKNOWN;
    nvmlReturn_t result = function(device_count);
    if (result == NVML_SUCCESS && *device_count > visible_device_limit()) {
        *device_count = visible_device_limit();
    }
    return result;
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int *device_count) {
    return filtered_count("nvmlDeviceGetCount", device_count);
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *device_count) {
    return filtered_count("nvmlDeviceGetCount_v2", device_count);
}

FORWARD_NVML(nvmlDeviceGetHandleByIndex,
             (unsigned int index, nvmlDevice_t *device),
             (index, device))
FORWARD_NVML(nvmlDeviceGetHandleByIndex_v2,
             (unsigned int index, nvmlDevice_t *device),
             (index, device))
FORWARD_NVML(nvmlDeviceGetHandleByPciBusId,
             (const char *pci_bus_id, nvmlDevice_t *device),
             (pci_bus_id, device))
FORWARD_NVML(nvmlDeviceGetHandleByPciBusId_v2,
             (const char *pci_bus_id, nvmlDevice_t *device),
             (pci_bus_id, device))
FORWARD_NVML(nvmlDeviceGetIndex,
             (nvmlDevice_t device, unsigned int *index),
             (device, index))
FORWARD_NVML(nvmlDeviceGetNvLinkState,
             (nvmlDevice_t device, unsigned int link, nvmlEnableState_t *is_active),
             (device, link, is_active))
FORWARD_NVML(nvmlDeviceGetNvLinkRemotePciInfo,
             (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci),
             (device, link, pci))
FORWARD_NVML(nvmlDeviceGetNvLinkRemotePciInfo_v2,
             (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci),
             (device, link, pci))
FORWARD_NVML(nvmlDeviceGetNvLinkRemoteDeviceType,
             (nvmlDevice_t device, unsigned int link,
              nvmlIntNvLinkDeviceType_t *device_type),
             (device, link, device_type))
FORWARD_NVML(nvmlDeviceGetNvLinkCapability,
             (nvmlDevice_t device, unsigned int link,
              nvmlNvLinkCapability_t capability, unsigned int *result),
             (device, link, capability, result))
FORWARD_NVML(nvmlDeviceGetCudaComputeCapability,
             (nvmlDevice_t device, int *major, int *minor),
             (device, major, minor))
FORWARD_NVML(nvmlDeviceGetP2PStatus,
             (nvmlDevice_t device1, nvmlDevice_t device2,
              nvmlGpuP2PCapsIndex_t p2p_index, nvmlGpuP2PStatus_t *p2p_status),
             (device1, device2, p2p_index, p2p_status))
FORWARD_NVML(nvmlDeviceGetFieldValues,
             (nvmlDevice_t device, int values_count, nvmlFieldValue_t *values),
             (device, values_count, values))
FORWARD_NVML(nvmlDeviceGetComputeRunningProcesses,
             (nvmlDevice_t device, unsigned int *info_count,
              nvmlProcessInfo_v1_t *infos),
             (device, info_count, infos))
FORWARD_NVML(nvmlDeviceGetGpuFabricInfoV,
             (nvmlDevice_t device, nvmlGpuFabricInfoV_t *gpu_fabric_info),
             (device, gpu_fabric_info))

const char *nvmlErrorString(nvmlReturn_t result) {
    typedef const char *(*function_t)(nvmlReturn_t);
    function_t function = (function_t)real_symbol("nvmlErrorString");
    return function == NULL ? "NVML proxy could not load the real driver" : function(result);
}

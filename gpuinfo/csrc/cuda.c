#include "types.h"


int cudaPresent() {
    return 0;
}


int cudaGetDeviceCount(int* count) {
    return -1;
}

int cudaGetDeviceProps(int index, GpuProp* obj) {
    return -1;
}

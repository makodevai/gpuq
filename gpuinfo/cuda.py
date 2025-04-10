import os
import re
import sys
import subprocess
import shutil
from functools import cache
from dataclasses import dataclass
from typing import Any


_cuda_gpu_info = re.compile(
    r"(?:(?:[0-9]+)%|N/A).*?([0-9]+)MiB +/ +([0-9]+)MiB.*?([0-9]+)%"
)
_cuda_process_info = re.compile(
    r"^ *?\| +([0-9]+) +[^ ]+ +[^ ]+ +([0-9]+) +(G|C) +.*? ([0-9]+)MiB \|", re.MULTILINE
)


def is_windows():
    return sys.platform.lower().startswith("win")


@dataclass
class CudaRuntimeInfo:
    index: int

    @property
    def utilisation(self) -> int:
        return get_gpu_status(self.index).get("utilisation", -1)

    @property
    def used_memory(self) -> int:
        return get_gpu_status(self.index).get("used_memory", -1)

    @property
    def pids(self) -> list[int]:
        return get_gpu_status(self.index).get("pids", [])


@dataclass
class CudaRuntimeInfoMock(CudaRuntimeInfo):
    def __init__(
        self, index: int, utilisation: int, used_memory: int, pids: list[int]
    ) -> None:
        super().__init__(index)
        self.__utilisation = utilisation
        self.__used_memory = used_memory
        self.__pids = pids

    @property
    def utilisation(self) -> int:
        return self.__utilisation

    @property
    def used_memory(self) -> int:
        return self.__used_memory

    @property
    def pids(self) -> list[int]:
        return self.__pids


@cache
def _get_nvidia_smi_path() -> str | None:
    path = shutil.which("nvidia-smi")
    if path is None:
        if is_windows():
            paths = [r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
        else:
            paths = ["/usr/bin/nvidia-smi", "/opt/bin/nvidia-smi"]

        for candidate in paths:
            if os.path.exists(candidate):
                path = candidate
                break

    return path


def _get_num_gpus() -> int:
    path = _get_nvidia_smi_path()
    if path is None:
        return 0

    out = subprocess.check_output([path, "-L"]).decode("UTF-8").strip()
    return sum(1 for line in out.splitlines() if line.strip())


def get_gpu_status(gpu_index: int) -> dict[str, Any]:
    path = _get_nvidia_smi_path()
    if not path:
        return {}

    out = subprocess.check_output(path).decode("UTF-8")

    ret: dict[str, Any] = {}
    usage_matches = _cuda_gpu_info.findall(out)
    for idx, match in enumerate(usage_matches):
        if idx != gpu_index:
            continue

        usage = int(match[-1])
        curr_mem = int(match[-3])
        ret = {"utilisation": usage, "used_memory": curr_mem, "pids": []}

    beg = out.find("Processes")
    usage_matches = _cuda_process_info.findall(out, beg)

    for m in usage_matches:
        gpu = int(m[0])
        if gpu != gpu_index:
            continue
        pid = int(m[1])
        ret["pids"].append(pid)

    return ret


def get_cuda_info(gpu_idx: int) -> CudaRuntimeInfo | None:
    if gpu_idx < 0:
        return None

    gpus = _get_num_gpus()
    if gpu_idx >= gpus:
        return None

    return CudaRuntimeInfo(
        index=gpu_idx,
    )

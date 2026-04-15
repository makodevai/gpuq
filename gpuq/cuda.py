from dataclasses import dataclass


@dataclass
class CudaRuntimeInfo:
    index: int

    @property
    def utilisation(self) -> int:
        from . import C

        return int(C._nvml_utilisation(self.index))

    @property
    def used_memory(self) -> int:
        """Used memory in MiB."""
        from . import C

        return int(C._nvml_used_memory(self.index))

    @property
    def pids(self) -> list[int]:
        from . import C

        return C._nvml_pids(self.index)


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


def get_cuda_info(gpu_idx: int) -> CudaRuntimeInfo | None:
    if gpu_idx < 0:
        return None

    from . import C

    count = 0
    try:
        count = int(C.count())
    except RuntimeError:
        return None

    if gpu_idx >= count:
        return None

    return CudaRuntimeInfo(index=gpu_idx)

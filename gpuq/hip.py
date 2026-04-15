from dataclasses import dataclass


@dataclass
class HipRuntimeInfo:
    index: int
    gfx: str
    drm: int
    node_idx: int

    @property
    def utilisation(self) -> int:
        from . import C

        return int(C._amdsmi_utilisation(self.index))

    @property
    def used_memory(self) -> int:
        """Used memory in MiB."""
        from . import C

        return int(C._amdsmi_used_memory(self.index))

    @property
    def pids(self) -> list[int]:
        from . import C

        return C._amdsmi_pids(self.index)


@dataclass
class HipRuntimeInfoMock(HipRuntimeInfo):
    def __init__(
        self,
        index: int,
        gfx: str,
        drm: int,
        node_idx: int,
        pids: list[int],
        utilisation: int = 0,
        used_memory: int = 0,
    ) -> None:
        super().__init__(index, gfx, drm, node_idx)
        self.__pids = pids
        self.__utilisation = utilisation
        self.__used_memory = used_memory

    @property
    def utilisation(self) -> int:
        return self.__utilisation

    @property
    def used_memory(self) -> int:
        return self.__used_memory

    @property
    def pids(self) -> list[int]:
        return self.__pids


def get_hip_info(gpu_idx: int) -> HipRuntimeInfo | None:
    if gpu_idx < 0:
        return None

    from . import C

    try:
        count = int(C.count())
    except RuntimeError:
        return None

    if gpu_idx >= count:
        return None

    gfx = C._amdsmi_gfx(gpu_idx)
    drm = int(C._amdsmi_drm(gpu_idx))
    node_id = int(C._amdsmi_node_id(gpu_idx))

    return HipRuntimeInfo(
        index=gpu_idx,
        gfx=gfx,
        drm=drm,
        node_idx=node_id,
    )

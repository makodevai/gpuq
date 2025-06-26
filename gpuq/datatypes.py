from typing import Any, TYPE_CHECKING
from uuid import uuid5, UUID
from enum import IntFlag, auto

from .cuda import CudaRuntimeInfo
from .hip import HipRuntimeInfo


if TYPE_CHECKING:
    from .impl import Implementation


class Provider(IntFlag):
    CUDA = auto()
    HIP = auto()

    @staticmethod
    def any() -> "Provider":
        return Provider(0)

    @staticmethod
    def all() -> "Provider":
        return Provider.CUDA | Provider.HIP


class Properties:
    def __init__(
        self, cobj: Any, local_index: int | None, impl: "Implementation"
    ) -> None:
        self.cobj = cobj
        self.local_index = local_index
        self.impl = impl
        self._cuda_info: CudaRuntimeInfo | None = None
        self._hip_info: HipRuntimeInfo | None = None

    @property
    def ord(self) -> int:
        """Ordinal of the GPU, across all providers and devices. Specific to the gpuq package."""
        return self.cobj.ord  # type: ignore[no-any-return]

    @property
    def uuid(self) -> UUID:
        return UUID(self.cobj.uuid)

    @property
    def provider(self) -> Provider:
        """Which runtime provides the GPU. Currently CUDA or HIP"""
        return Provider[self.cobj.provider]

    @property
    def index(self) -> int | None:
        """Index of the GPU as seen by the calling process. This can be affected by
        various *_VISIBLE_DEVICES environment variables.

        Can be ``None`` if the GPU is not visible by this process. Also see ``is_visible``
        for more information.

        This index is provider-specific.

        > **Note:** visibility is determined at the moment of constructing the object and will not
        > reflect any changes made later.
        """
        return self.local_index

    @property
    def system_index(self) -> int:
        """System-wide index of the GPU, i.e., its index when ignoring *_VISIBLE_DEVICES.

        This index is provider-specific.

        > **Note:** system-wide index is determined by temporarily removing *_VISIBLE_DEVICES
        > variables.
        > This might cause race conditions if the variables are also used/modified by other
        > parts of the system at the same time. Please keep this in mind when using the package.
        """
        return self.cobj.index  # type: ignore[no-any-return]

    @property
    def is_visible(self) -> bool:
        """Whether the GPU is visible by the current process, per the relevant *_VISIBLE_DEVICES
        environment variables.

        > **Note:** visibility is determined at the moment of constructing the object and will not
        > reflect any changes made later.

        > **Note:** the implementation will temporarily remove any *_VISIBLE_DEVICES variables
        > when obtaining information about the GPU, to correctly report other properties.
        > This might cause race conditions if the variables are also used/modified by other
        > parts of the system at the same time. Please keep this in mind when using the package.
        """
        return self.index is not None

    @property
    def name(self) -> str:
        """Name of the GPU"""
        return self.cobj.name  # type: ignore[no-any-return]

    @property
    def short_name(self) -> str | None:
        """Short name of the GPU, if known. Otherwise None."""
        from .short_names import short_names

        return short_names.get(self.cobj.name, None)

    @property
    def major(self) -> int:
        return self.cobj.major  # type: ignore[no-any-return]

    @property
    def minor(self) -> int:
        return self.cobj.minor  # type: ignore[no-any-return]

    @property
    def total_memory(self) -> int:
        return self.cobj.total_memory  # type: ignore[no-any-return]

    @property
    def sms_count(self) -> int:
        return self.cobj.sms_count  # type: ignore[no-any-return]

    @property
    def sm_threads(self) -> int:
        return self.cobj.sm_threads  # type: ignore[no-any-return]

    @property
    def sm_shared_memory(self) -> int:
        return self.cobj.sm_shared_memory  # type: ignore[no-any-return]

    @property
    def sm_registers(self) -> int:
        return self.cobj.sm_registers  # type: ignore[no-any-return]

    @property
    def sm_blocks(self) -> int:
        return self.cobj.sm_blocks  # type: ignore[no-any-return]

    @property
    def block_threads(self) -> int:
        return self.cobj.block_threads  # type: ignore[no-any-return]

    @property
    def block_shared_memory(self) -> int:
        return self.cobj.block_shared_memory  # type: ignore[no-any-return]

    @property
    def block_registers(self) -> int:
        return self.cobj.block_registers  # type: ignore[no-any-return]

    @property
    def warp_size(self) -> int:
        return self.cobj.warp_size  # type: ignore[no-any-return]

    @property
    def l2_cache_size(self) -> int:
        return self.cobj.l2_cache_size  # type: ignore[no-any-return]

    @property
    def concurrent_kernels(self) -> bool:
        return self.cobj.concurrent_kernels  # type: ignore[no-any-return]

    @property
    def async_engines_count(self) -> int:
        return self.cobj.async_engines_count  # type: ignore[no-any-return]

    @property
    def cooperative(self) -> bool:
        return self.cobj.cooperative  # type: ignore[no-any-return]

    @property
    def cuda_info(self) -> CudaRuntimeInfo | None:
        if self._cuda_info is not None:
            return self._cuda_info

        if self.provider != Provider.CUDA:
            return None

        self._cuda_info = self.impl.cuda_runtime_info(self.system_index)
        return self._cuda_info

    @property
    def hip_info(self) -> HipRuntimeInfo | None:
        if self._hip_info is not None:
            return self._hip_info

        if self.provider != Provider.HIP:
            return None

        self._hip_info = self.impl.hip_runtime_info(self.system_index)
        return self._hip_info

    def asdict(self, strip_index: bool = False) -> dict[str, Any]:
        ret = {
            "ord": self.ord,
            "uuid": self.uuid.hex,
            "provider": self.provider.name,
            "index": self.index,
            "system_index": self.system_index,
            "name": self.name,
            "major": self.major,
            "minor": self.minor,
            "total_memory": self.total_memory,
            "sms_count": self.sms_count,
            "sm_threads": self.sm_threads,
            "sm_shared_memory": self.sm_shared_memory,
            "sm_registers": self.sm_registers,
            "sm_blocks": self.sm_blocks,
            "block_threads": self.block_threads,
            "block_shared_memory": self.block_shared_memory,
            "block_registers": self.block_registers,
            "warp_size": self.warp_size,
            "l2_cache_size": self.l2_cache_size,
            "concurrent_kernels": self.concurrent_kernels,
            "async_engines_count": self.async_engines_count,
            "cooperative": self.cooperative,
        }

        if strip_index:
            del ret["ord"]
            del ret["uuid"]
            del ret["index"]
            del ret["system_index"]

        return ret

    def __eq__(self, other: Any) -> bool:
        """Return True if self and other are equivalent GPUs.

        If you want to check if two GPus are the same physical devices,
        simply compare their indices.
        """
        if not isinstance(other, Properties):
            return False
        if self.impl is other.impl and self.index == other.index:
            return True
        return self.asdict(strip_index=True) == other.asdict(strip_index=True)

    def __str__(self) -> str:
        props = {"uuid": str(self.uuid)}
        props.update(self.asdict(strip_index=True))
        del props["provider"]
        del props["name"]
        return (
            self.__repr__()
            + "{\n    "
            + "\n    ".join(f"{key}: {value}" for key, value in props.items())
            + "\n}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}({self.provider.name}[{self.system_index} -> {self.index}], {self.name!r})"

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["cobj"] = None
        state["_ord"] = self.ord
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        ord = state.pop("_ord")
        self.__dict__.update(state)
        with self.impl.save_visible(clear=True):
            try:
                self.cobj = self.impl.c_get(ord)
            except Exception as exp:
                raise RuntimeError(
                    f"Failed to unpickle GPU properties object with global ID {ord}"
                ) from exp


class MockCObj:
    def __init__(
        self,
        ord: int = 0,
        uuid: str | None = None,
        provider: str = "CUDA",
        index: int = 0,
        name: str = "{} Mock Device",
        major: int = 1,
        minor: int = 2,
        total_memory: int = 8 * 1024**3,
        sms_count: int = 12,
        sm_threads: int = 2048,
        sm_shared_memory: int = 16 * 1024,
        sm_registers: int = 512,
        sm_blocks: int = 4,
        block_threads: int = 1024,
        block_shared_memory: int = 8 * 1024,
        block_registers: int = 256,
        warp_size: int = 32,
        l2_cache_size: int = 8 * 1024**2,
        concurrent_kernels: bool = True,
        async_engines_count: int = 0,
        cooperative: bool = True,
    ) -> None:
        name = name.format(provider)

        if uuid is None:
            args = dict(locals())
            del args["self"]
            del args["uuid"]
            args_str = str(args)
            # namespace is hex of (zero-padded) "mako_gpuq_mock"
            uuid = uuid5(UUID("6d616b6f-5f67-7075-715f-6d6f636B0000"), args_str).hex

        self.ord = ord
        self.uuid = uuid
        self.provider = provider
        self.index = index
        self.name = name
        self.major = major
        self.minor = minor
        self.total_memory = total_memory
        self.sms_count = sms_count
        self.sm_threads = sm_threads
        self.sm_shared_memory = sm_shared_memory
        self.sm_registers = sm_registers
        self.sm_blocks = sm_blocks
        self.block_threads = block_threads
        self.block_shared_memory = block_shared_memory
        self.block_registers = block_registers
        self.warp_size = warp_size
        self.l2_cache_size = l2_cache_size
        self.concurrent_kernels = concurrent_kernels
        self.async_engines_count = async_engines_count
        self.cooperative = cooperative

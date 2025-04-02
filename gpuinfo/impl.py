import os
from typing import Any, ContextManager, Generator
from contextlib import contextmanager

from . import C
from .provider import Provider


Visible = dict[Provider, list[int] | None]


class Implementation():
    def provider_check(self, provider: Provider) -> bool: ...
    def save_visible(self, clear: bool = True) -> ContextManager[Visible]: ...
    def count(self) -> int: ...
    def get(self, ord: int) -> Any: ...


class GenuineImplementation(Implementation):
    def provider_check(self, provider: Provider) -> bool:
        return {
            Provider.CUDA: (lambda: C.checkcuda() == 0),
            Provider.HIP: (lambda: C.checkamd() == 0),
        }[provider]()

    @contextmanager
    def save_visible(self, clear: bool = True) -> Generator[Visible]:
        cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        hip = os.environ.get('HIP_VISIBLE_DEVICES', None)

        def _is_int(value, _prefix):
            try:
                value = int(value)
                return True
            except:
                raise ValueError(f'{_prefix}_VISIBLE_DEVICES environment variable contains values that are not integer - this is currently not supported: {value!r}') from None

        if cuda is not None:
            parsed_cuda = { int(g) for g in cuda.split(',') if _is_int(g, 'CUDA') }
            parsed_cuda = sorted(list(parsed_cuda))
        else:
            parsed_cuda = None

        if hip is not None:
            parsed_hip = { int(g) for g in hip.split(',') if _is_int(g, 'HIP') }
            parsed_hip = sorted(list(parsed_hip))
        else:
            parsed_hip = parsed_cuda

        if clear:
            if cuda is not None:
                del os.environ['CUDA_VISIBLE_DEVICES']
            if hip is not None:
                del os.environ['HIP_VISIBLE_DEVICES']

        try:
            yield { Provider.CUDA: parsed_cuda, Provider.HIP: parsed_hip }
        finally:
            if clear:
                if cuda is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
                if hip is not None:
                    os.environ['HIP_VISIBLE_DEVICES'] = hip

    def count(self) -> int:
        return C.count()

    def get(self, ord: int) -> Any:
        return C.get(ord)


class MockImplementation(Implementation):
    def __init__(self,
            cuda_count: int | None = 1,
            hip_count: int | None = None,
            cuda_visible: list[int] | None = None,
            hip_visible: list[int] | None = None,
            name: str = "MockDevice",
            major: int = 1,
            minor: int = 2,
            total_memory: int = 8*1024**3,
            sms_count: int = 12,
            sm_threads: int = 2048,
            sm_shared_memory: int = 16*1024,
            sm_registers: int = 512,
            sm_blocks: int = 4,
            block_threads: int = 1024,
            block_shared_memory: int = 8*1024,
            block_registers: int = 256,
            warp_size: int = 32,
            l2_cache_size: int = 8*1024**2,
            concurrent_kernels: bool = True,
            async_engines_count: int = 0,
            cooperative: bool = True,
        ) -> None:
        if (cuda_count is not None and cuda_count < 0) or (hip_count is not None and hip_count < 0):
            raise ValueError('Negative number of mock devices!')

        self.cuda_count = cuda_count
        self.hip_count = hip_count
        self.overall_count = (cuda_count or 0) + (hip_count or 0)

        self.cuda_visible = None
        self.hip_visible = None

        self.cobj_args = {
            'name': name,
            'major': major,
            'minor': minor,
            'total_memory': total_memory,
            'sms_count': sms_count,
            'sm_threads': sm_threads,
            'sm_shared_memory': sm_shared_memory,
            'sm_registers': sm_registers,
            'sm_blocks': sm_blocks,
            'block_threads': block_threads,
            'block_shared_memory': block_shared_memory,
            'block_registers': block_registers,
            'warp_size': warp_size,
            'l2_cache_size': l2_cache_size,
            'concurrent_kernels': concurrent_kernels,
            'async_engines_count': async_engines_count,
            'cooperative': cooperative,
        }

        if cuda_visible is not None:
            self.cuda_visible = [idx for idx in range(self.cuda_count or 0) if idx in cuda_visible]
        else:
            self.cuda_visible = list(range(self.cuda_count or 0))

        if hip_visible is not None:
            self.hip_visible = [idx for idx in range(self.hip_count or 0) if idx in hip_visible]
        else:
            self.hip_visible = list(range(self.hip_count or 0))

    def provider_check(self, provider: Provider) -> bool:
        return {
            Provider.CUDA: (lambda: self.cuda_count is not None),
            Provider.HIP: (lambda: self.hip_count is not None),
        }[provider]()

    @contextmanager
    def save_visible(self, clear: bool = True) -> Generator[Visible]:
        cuda = self.cuda_visible.copy()
        hip = self.hip_visible.copy()

        if clear:
            self.cuda_visible = list(range(self.cuda_count or 0))
            self.hip_visible = list(range(self.hip_count or 0))

        try:
            yield { Provider.CUDA: cuda, Provider.HIP: hip }
        finally:
            if clear:
                self.cuda_visible = cuda
                self.hip_visible = hip

    def count(self) -> int:
        return len(self.cuda_visible) + len(self.hip_visible)

    def get(self, ord: int) -> Any:
        if ord < 0 or ord >= self.overall_count:
            raise IndexError('Invalid device index')

        from .mock import MockCObj
        index = ord
        provider = 'CUDA'
        if ord >= (self.cuda_count or 0):
            index = ord - (self.cuda_count or 0)
            provider = 'HIP'

        return MockCObj(ord=ord, provider=provider, index=index, **self.cobj_args)

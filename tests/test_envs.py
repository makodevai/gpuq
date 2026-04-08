import os
from contextlib import contextmanager
from typing import Generator

import gpuq as G


@contextmanager
def env_overwrite(**kwargs: str) -> Generator[None, None, None]:
    env = os.environ.copy()
    os.environ.update(kwargs)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env)


def test_empty_env() -> None:
    with env_overwrite(CUDA_VISIBLE_DEVICES=""):
        assert not G.query(G.Provider.CUDA, visible_only=True)


def test_empty_env_hip() -> None:
    with env_overwrite(HIP_VISIBLE_DEVICES=""):
        assert not G.query(G.Provider.HIP, visible_only=True)


def test_empty_env_mock() -> None:
    with env_overwrite(CUDA_VISIBLE_DEVICES=""):
        with G.mock(cuda_count=1, hip_count=0):
            assert not G.query(visible_only=True)


def test_empty_env_hip_mock() -> None:
    with env_overwrite(HIP_VISIBLE_DEVICES=""):
        with G.mock(cuda_count=0, hip_count=1):
            assert not G.query(visible_only=True)


def test_count_visible_only_after_false() -> None:
    """Regression: count(visible_only=True) must not be affected by prior count(visible_only=False)."""
    with env_overwrite(CUDA_VISIBLE_DEVICES="1"):
        with G.mock(cuda_count=8):
            assert G.count(visible_only=False) == 8
            assert G.count(visible_only=True) == 1


def test_count_false_after_visible_only() -> None:
    """Regression: count(visible_only=False) must not be affected by prior count(visible_only=True)."""
    with env_overwrite(CUDA_VISIBLE_DEVICES="1"):
        with G.mock(cuda_count=8):
            assert G.count(visible_only=True) == 1
            assert G.count(visible_only=False) == 8


def test_count_ordering_multiple_visible() -> None:
    """Count ordering with multiple visible devices — no state leakage."""
    with env_overwrite(CUDA_VISIBLE_DEVICES="1,3,5"):
        with G.mock(cuda_count=8):
            assert G.count(visible_only=False) == 8
            assert G.count(visible_only=True) == 3
            assert G.count(visible_only=False) == 8
            assert G.count(visible_only=True) == 3


def test_index_mapping_multiple_visible() -> None:
    """query() maps system indices to correct local indices with multiple visible."""
    with env_overwrite(CUDA_VISIBLE_DEVICES="2,5"):
        with G.mock(cuda_count=8):
            visible = G.query(visible_only=True)
            assert len(visible) == 2
            assert visible[0].index == 0
            assert visible[0].system_index == 2
            assert visible[1].index == 1
            assert visible[1].system_index == 5


def test_index_mapping_all_gpus_visibility() -> None:
    """query(visible_only=False) returns all GPUs with correct visibility annotations."""
    with env_overwrite(CUDA_VISIBLE_DEVICES="1,4"):
        with G.mock(cuda_count=6):
            all_gpus = G.query(visible_only=False)
            assert len(all_gpus) == 6

            for gpu in all_gpus:
                if gpu.system_index in (1, 4):
                    assert gpu.is_visible
                    assert gpu.index is not None
                else:
                    assert not gpu.is_visible
                    assert gpu.index is None

            visible = [g for g in all_gpus if g.is_visible]
            assert visible[0].index == 0
            assert visible[0].system_index == 1
            assert visible[1].index == 1
            assert visible[1].system_index == 4


def test_count_ordering_mixed_providers() -> None:
    """Count ordering with both CUDA and HIP visible subsets."""
    with env_overwrite(CUDA_VISIBLE_DEVICES="0,2", HIP_VISIBLE_DEVICES="1"):
        with G.mock(cuda_count=4, hip_count=4):
            assert G.count(visible_only=False) == 8
            assert G.count(visible_only=True) == 3
            assert G.count(G.Provider.CUDA, visible_only=True) == 2
            assert G.count(G.Provider.HIP, visible_only=True) == 1
            assert G.count(visible_only=False) == 8


def test_index_mapping_hip() -> None:
    """HIP index mapping with multiple visible devices."""
    with env_overwrite(HIP_VISIBLE_DEVICES="0,3"):
        with G.mock(cuda_count=None, hip_count=6):
            visible = G.query(G.Provider.HIP, visible_only=True)
            assert len(visible) == 2
            assert visible[0].index == 0
            assert visible[0].system_index == 0
            assert visible[1].index == 1
            assert visible[1].system_index == 3

            all_gpus = G.query(G.Provider.HIP, visible_only=False)
            assert len(all_gpus) == 6
            for gpu in all_gpus:
                if gpu.system_index in (0, 3):
                    assert gpu.is_visible
                    assert gpu.index is not None
                else:
                    assert not gpu.is_visible
                    assert gpu.index is None

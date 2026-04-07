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

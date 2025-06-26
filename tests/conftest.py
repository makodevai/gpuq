from threading import local
from typing import Generator

import pytest
import gpuq


@pytest.fixture(autouse=True)
def run_around_tests() -> Generator[None, None, None]:
    assert gpuq._default_impl is None
    yield
    gpuq._default_impl = None
    gpuq._current_implementation = local()

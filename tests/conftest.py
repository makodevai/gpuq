from threading import local
from typing import Generator

import pytest
import gpuinfo


@pytest.fixture(autouse=True)
def run_around_tests() -> Generator[None, None, None]:
    assert gpuinfo._default_impl is None
    yield
    gpuinfo._default_impl = None
    gpuinfo._current_implementation = local()

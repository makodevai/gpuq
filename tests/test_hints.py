import pytest

import gpuq.impl
import gpuq.C


max_hints = 16
max_hint_len = 127


@pytest.fixture(autouse=True)
def restore_hints():
    try:
        yield
    finally:
        gpuq.impl._restore_default_hints()



def test_just_right_hint():
    gpuq.C._set_location_hints([b"t" * max_hint_len])


def test_just_right_hints():
    gpuq.C._set_location_hints([b"t" * max_hint_len] * max_hints)


def test_no_args():
    with pytest.raises(TypeError):
        gpuq.C._set_location_hints()  # type: ignore


def test_too_many_args():
    with pytest.raises(TypeError):
        gpuq.C._set_location_hints(b"/test", b"/foo")  # type: ignore


def test_keyword_arg():
    with pytest.raises(TypeError):
        gpuq.C._set_location_hints(locs=[b"/test"])  # type: ignore


def test_wrong_type():
    with pytest.raises(TypeError):
        gpuq.C._set_location_hints(tuple(b"/test" for _ in range(2)))  # type: ignore


def test_wrong_element_type():
    with pytest.raises(TypeError):
        gpuq.C._set_location_hints(["/test"])  # type: ignore


def test_too_many_hints():
    with pytest.raises(ValueError):
        gpuq.C._set_location_hints([b"/test" for _ in range(max_hints + 1)])


def test_one_too_long():
    with pytest.raises(ValueError):
        gpuq.C._set_location_hints([
            b"/test"
            b"t" * (max_hint_len + 1)
        ])  # type: ignore

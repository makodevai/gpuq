import os
import gpuq as G
from gpuq import _set_impl, _with_impl, _get_impl
from gpuq.impl import GenuineImplementation, MockImplementation


def test_default_is_mock() -> None:
    assert isinstance(G.default_impl, GenuineImplementation)

    G._default_impl = None
    os.environ["MAKO_MOCK_GPU"] = "1"
    try:
        assert isinstance(G.default_impl, MockImplementation)
    finally:
        del os.environ["MAKO_MOCK_GPU"]


def test_default_impl() -> None:
    assert _get_impl() is G.default_impl


def test_set_impl() -> None:
    impl2 = GenuineImplementation()
    assert _set_impl(impl2) is G.default_impl
    assert _get_impl() is impl2

    assert _set_impl(None) is impl2
    assert _get_impl() is G.default_impl


def test_set_impl_obj() -> None:
    impl2 = GenuineImplementation()
    assert impl2.set() is G.default_impl
    assert _get_impl() is impl2

    assert G.default_impl.set() is impl2
    assert _get_impl() is G.default_impl


def test_with_impl() -> None:
    impl2 = MockImplementation()
    with _with_impl(impl2) as ret:
        assert ret is impl2
        assert _get_impl() is impl2

    assert _get_impl() is G.default_impl


def test_with_impl_obj() -> None:
    impl2 = MockImplementation()
    with impl2 as ret:
        assert ret is impl2
        assert _get_impl() is impl2

    assert _get_impl() is G.default_impl


def test_nested_with() -> None:
    impl2 = MockImplementation()
    impl3 = GenuineImplementation()

    assert _get_impl() is G.default_impl
    with _with_impl(impl2):
        assert _get_impl() is impl2
        with impl3:
            assert _get_impl() is impl3
            with G.default_impl:
                assert _get_impl() is G.default_impl
                with _with_impl(impl2):
                    assert _get_impl() is impl2
                    with _with_impl(None):
                        assert _get_impl() is G.default_impl
                    assert _get_impl() is impl2
                assert _get_impl() is G.default_impl
            assert _get_impl() is impl3
        assert _get_impl() is impl2
    assert _get_impl() is G.default_impl


def test_obj_api_count() -> None:
    impl2 = MockImplementation(cuda_count=1, hip_count=None)
    impl3 = MockImplementation(cuda_count=None, hip_count=2)

    assert impl2.count() == G.count(impl=impl2)
    assert impl3.count() == G.count(impl=impl3)

    assert impl2.count() != impl3.count()

    with impl2:
        assert G.count() == impl2.count()
        assert G.count() != impl3.count()

    with impl3:
        assert G.count() != impl2.count()
        assert G.count() == impl3.count()


def test_obj_api_count_visible() -> None:
    impl = MockImplementation(cuda_count=8, cuda_visible=[0, 1, 2])
    assert impl.count(visible_only=False) == 8
    assert impl.count(visible_only=True) == 3

    assert G.count(visible_only=False, impl=impl) == impl.count(visible_only=False)
    assert G.count(visible_only=True, impl=impl) == impl.count(visible_only=True)


def test_obj_api_get() -> None:
    impl2 = MockImplementation(cuda_count=1, hip_count=None)
    impl3 = MockImplementation(cuda_count=None, hip_count=2)

    assert impl2.get(0) == G.get(0, impl=impl2)
    assert impl3.get(0) == G.get(0, impl=impl3)
    assert impl2.get(0) != impl3.get(0)

    with impl2:
        assert G.get(0) == impl2.get(0)
        assert G.get(0) != impl3.get(0)

    with impl3:
        assert G.get(0) != impl2.get(0)
        assert G.get(0) == impl3.get(0)


def test_obj_api_query() -> None:
    impl2 = MockImplementation(cuda_count=1, hip_count=None)
    impl3 = MockImplementation(cuda_count=None, hip_count=2)

    assert impl2.query() == G.query(impl=impl2)
    assert impl3.query() == G.query(impl=impl3)
    assert impl2.query() != impl3.query()

    with impl2:
        assert G.query() == impl2.query()
        assert G.query() != impl3.query()

    with impl3:
        assert G.query() != impl2.query()
        assert G.query() == impl3.query()

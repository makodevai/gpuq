import gpuinfo as G
from gpuinfo import _default_impl, _set_impl, _with_impl, _get_impl, GenuineImplementation, MockImplementation


def test_default_impl():
    assert _get_impl() is _default_impl


def test_set_impl():
    impl2 = GenuineImplementation()
    assert _set_impl(impl2) is _default_impl
    assert _get_impl() is impl2

    assert _set_impl(None) is impl2
    assert _get_impl() is _default_impl


def test_set_impl_obj():
    impl2 = GenuineImplementation()
    assert impl2.set() is _default_impl
    assert _get_impl() is impl2

    assert _default_impl.set() is impl2
    assert _get_impl() is _default_impl


def test_with_impl():
    impl2 = MockImplementation()
    with _with_impl(impl2) as ret:
        assert ret is impl2
        assert _get_impl() is impl2

    assert _get_impl() is _default_impl


def test_with_impl_obj():
    impl2 = MockImplementation()
    with impl2 as ret:
        assert ret is impl2
        assert _get_impl() is impl2

    assert _get_impl() is _default_impl


def test_nested_with():
    impl2 = MockImplementation()
    impl3 = GenuineImplementation()

    assert _get_impl() is _default_impl
    with _with_impl(impl2):
        assert _get_impl() is impl2
        with impl3:
            assert _get_impl() is impl3
            with _default_impl:
                assert _get_impl() is _default_impl
                with _with_impl(impl2):
                    assert _get_impl() is impl2
                    with _with_impl(None):
                        assert _get_impl() is _default_impl
                    assert _get_impl() is impl2
                assert _get_impl() is _default_impl
            assert _get_impl() is impl3
        assert _get_impl() is impl2
    assert _get_impl() is _default_impl


def test_obj_api_count():
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


def test_obj_api_count_visible():
    impl = MockImplementation(cuda_count=8, cuda_visible=[0, 1, 2])
    assert impl.count(visible_only=False) == 8
    assert impl.count(visible_only=True) == 3

    assert G.count(visible_only=False, impl=impl) == impl.count(visible_only=False)
    assert G.count(visible_only=True, impl=impl) == impl.count(visible_only=True)



def test_obj_api_get():
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


def test_obj_api_query():
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

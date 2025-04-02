import pytest

import gpuinfo as G


def test_runtimes():
    with G.mock(cuda_count=0, hip_count=0):
        assert G.hasamd()
        assert G.hascuda()

    with G.mock(cuda_count=None, hip_count=None):
        assert not G.hasamd()
        assert not G.hascuda()


def test_count():
    with G.mock(cuda_count=2, hip_count=3):
        assert G.count() == 5
        assert G.count(G.Provider.any()) == 5
        assert G.count(G.Provider.all()) == 5
        assert G.count(G.Provider.CUDA) == 2
        assert G.count(G.Provider.HIP) == 3


def test_simple_query():
    with G.mock(cuda_count=1, hip_count=1):
        assert G.count() == 2
        assert G.get(0).provider == G.Provider.CUDA
        assert G.get(1).provider == G.Provider.HIP

        all = G.query()
        assert len(all) == 2
        assert all[0].provider == G.Provider.CUDA
        assert all[1].provider == G.Provider.HIP

        cuda = G.query(G.Provider.CUDA)
        assert len(cuda) == 1
        assert cuda[0].provider == G.Provider.CUDA

        hip = G.query(G.Provider.HIP)
        assert len(hip) == 1
        assert hip[0].provider == G.Provider.HIP


def test_get_visible():
    with G.mock(cuda_count=2, cuda_visible=[1]):
        assert G.count(visible_only=False) == 2
        assert G.count(visible_only=True) == 1

        g1 = G.get(0, visible_only=False)
        g2 = G.get(1, visible_only=False)

        assert not g1.is_visible
        assert g1.local_index is None
        assert g1.system_index == 0

        assert g2.is_visible
        assert g2.local_index == 0
        assert g2.system_index == 1

        assert g2.ord == G.get(0, visible_only=True).ord

        with pytest.raises(IndexError):
            G.get(1, visible_only=True)


def test_visible_hip_from_cuda():
    with G.mock(cuda_count=None, hip_count=2, cuda_visible=[0]):
        # This might be a bit counter-intuitive, but follows the HIP behaviour
        # that if CUDA_VISIBLE_DEVICES is set but HIP_VISIBLE_DEVICES is not,
        # then the CUDA variable is inherited
        assert G.count(visible_only=True) == 1

    with G.mock(cuda_count=None, hip_count=2, cuda_visible=[0], hip_visible=[0,1]):
        assert G.count(visible_only=True) == 2


def test_visible_hip_from_cuda_2():
    with G.mock(cuda_count=2, hip_count=2, cuda_visible=[1]):
        assert G.count(visible_only=False) == 4
        assert G.count(visible_only=True) == 2 # see test case above why

        assert G.count(G.Provider.CUDA, visible_only=False) == 2
        assert G.count(G.Provider.HIP, visible_only=False) == 2

        assert G.count(G.Provider.CUDA, visible_only=True) == 1
        assert G.count(G.Provider.HIP, visible_only=True) == 1

        cuda = G.get(0, G.Provider.CUDA, visible_only=True)
        hip = G.get(0, G.Provider.HIP, visible_only=True)
        assert hip.local_index == cuda.local_index
        assert hip.system_index == cuda.system_index


def test_query_filtering():
    with G.mock(cuda_count=1, hip_count=0):
        assert G.query(G.Provider.any())
        assert G.query(G.Provider.CUDA)

        assert G.query(G.Provider.HIP) == []
        assert G.query(G.Provider.HIP, required=G.Provider.CUDA) == []

        with pytest.raises(RuntimeError):
            G.query(G.Provider.HIP, required=G.Provider.HIP)
        with pytest.raises(RuntimeError):
            G.query(G.Provider.HIP, required=True)
        with pytest.raises(RuntimeError):
            G.query(G.Provider.CUDA, required=G.Provider.HIP)

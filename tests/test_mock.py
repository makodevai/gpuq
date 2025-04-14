import pytest
import multiprocessing as mp
import multiprocessing.queues as mp_q

import gpuinfo as G


def test_runtimes() -> None:
    with G.mock(cuda_count=0, hip_count=0):
        assert G.hasamd()
        assert G.hascuda()

    with G.mock(cuda_count=None, hip_count=None):
        assert not G.hasamd()
        assert not G.hascuda()


def test_count() -> None:
    with G.mock(cuda_count=2, hip_count=3):
        assert G.count() == 5
        assert G.count(G.Provider.any()) == 5
        assert G.count(G.Provider.all()) == 5
        assert G.count(G.Provider.CUDA) == 2
        assert G.count(G.Provider.HIP) == 3


def test_simple_query() -> None:
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


def test_get_visible() -> None:
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


def test_visible_hip_from_cuda() -> None:
    with G.mock(cuda_count=None, hip_count=2, cuda_visible=[0]):
        # This might be a bit counter-intuitive, but follows the HIP behaviour
        # that if CUDA_VISIBLE_DEVICES is set but HIP_VISIBLE_DEVICES is not,
        # then the CUDA variable is inherited
        assert G.count(visible_only=True) == 1

    with G.mock(cuda_count=None, hip_count=2, cuda_visible=[0], hip_visible=[0,1]):
        assert G.count(visible_only=True) == 2


def test_visible_hip_from_cuda_2() -> None:
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


def test_query_filtering() -> None:
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


def test_default_names() -> None:
    with G.mock(cuda_count=1):
        assert G.get(0).name == 'CUDA Mock Device'

    with G.mock(cuda_count=None, hip_count=1):
        assert G.get(0).name == 'HIP Mock Device'

    with G.mock(cuda_count=1, hip_count=1):
        assert G.get(0).name == 'CUDA Mock Device'
        assert G.get(1).name == 'HIP Mock Device'


def test_uuid_uniqueness() -> None:
    with G.mock(cuda_count=16, hip_count=None):
        gpus = G.query()
        uuids_nvidia = { gpu.uuid for gpu in gpus }

    with G.mock(cuda_count=None, hip_count=16):
        gpus = G.query()
        uuids_amd = { gpu.uuid for gpu in gpus }

    assert len(uuids_nvidia) == 16
    assert len(uuids_amd) == 16
    assert not uuids_nvidia.intersection(uuids_amd)


def test_uuid_consistency() -> None:
    with G.mock(cuda_count=16, hip_count=None):
        gpus = G.query()
        uuids_nvidia_1 = [gpu.uuid for gpu in gpus]

    with G.mock(cuda_count=16, hip_count=None):
        gpus = G.query()
        uuids_nvidia_2 = [gpu.uuid for gpu in gpus]

    assert uuids_nvidia_1 == uuids_nvidia_2


def test_uuid_args() -> None:
    with G.mock(cuda_count=1, total_memory=128):
        uuid1 = G.get(0).uuid

    with G.mock(cuda_count=1, total_memory=128):
        uuid2 = G.get(0).uuid

    with G.mock(cuda_count=1, total_memory=129):
        uuid3 = G.get(0).uuid

    assert uuid1 == uuid2
    assert uuid1 != uuid3


def test_hip_runtime() -> None:
    with G.mock(cuda_count=None, hip_count=1,
                hip_drm=128,
                hip_gfx="942",
                hip_node_idx=2,
                hip_pids=[1,1024]):
        gpu = G.get(0)
        assert gpu.cuda_info is None
        assert gpu.hip_info is not None

        assert gpu.hip_info.drm == 128
        assert gpu.hip_info.gfx == "942"
        assert gpu.hip_info.node_idx == 2
        assert gpu.hip_info.pids == [1, 1024]


def test_cuda_runtime() -> None:
    with G.mock(cuda_count=1, hip_count=None,
                cuda_utilisation=11,
                cuda_memory=1552,
                cuda_pids=[1, 1024]):
        gpu = G.get(0)
        assert gpu.cuda_info is not None
        assert gpu.hip_info is None

        assert gpu.cuda_info.utilisation == 11
        assert gpu.cuda_info.used_memory == 1552
        assert gpu.cuda_info.pids == [1, 1024]


def _child(gpu: G.Properties, queue: mp_q.Queue[G.Properties]) -> None:
    queue.put(gpu)


def test_mp() -> None:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with G.mock(cuda_count=1):
        gpu = G.get(0)
        proc = ctx.Process(target=_child, args=(gpu, q))
        proc.start()
        proc.join(2)
        gpu2 = q.get_nowait()
        assert gpu == gpu2
        assert gpu.ord == gpu2.ord
        assert gpu.uuid == gpu2.uuid

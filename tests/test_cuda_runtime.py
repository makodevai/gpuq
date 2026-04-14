import gpuq
from gpuq.cuda import CudaRuntimeInfoMock, get_cuda_info


def test_cuda_runtime_info_mock() -> None:
    mock = CudaRuntimeInfoMock(
        index=0, utilisation=42, used_memory=1024, pids=[100, 200]
    )
    assert mock.index == 0
    assert mock.utilisation == 42
    assert mock.used_memory == 1024
    assert mock.pids == [100, 200]


def test_cuda_runtime_info_via_mock_impl() -> None:
    impl = gpuq.mock(cuda_count=2, cuda_utilisation=55, cuda_memory=512, cuda_pids=[1, 2])
    with impl:
        g = gpuq.get(0, visible_only=False)
        info = g.cuda_info
        assert info is not None
        assert info.utilisation == 55
        assert info.used_memory == 512
        assert info.pids == [1, 2]


def test_get_cuda_info_out_of_range() -> None:
    assert get_cuda_info(-1) is None

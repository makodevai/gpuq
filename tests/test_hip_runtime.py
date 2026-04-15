import gpuq
from gpuq.hip import HipRuntimeInfoMock


def test_hip_runtime_info_mock() -> None:
    mock = HipRuntimeInfoMock(
        index=0,
        gfx="942",
        drm=128,
        node_idx=2,
        pids=[100, 200],
        utilisation=75,
        used_memory=4096,
    )
    assert mock.index == 0
    assert mock.gfx == "942"
    assert mock.drm == 128
    assert mock.node_idx == 2
    assert mock.utilisation == 75
    assert mock.used_memory == 4096
    assert mock.pids == [100, 200]


def test_hip_runtime_info_via_mock_impl() -> None:
    impl = gpuq.mock(
        hip_count=2,
        cuda_count=None,
        hip_gfx="942",
        hip_drm=128,
        hip_node_idx=2,
        hip_pids=[1, 2],
        hip_utilisation=50,
        hip_memory=2048,
    )
    with impl:
        g = gpuq.get(0, visible_only=False)
        info = g.hip_info
        assert info is not None
        assert info.gfx == "942"
        assert info.drm == 128
        assert info.node_idx == 2
        assert info.utilisation == 50
        assert info.used_memory == 2048
        assert info.pids == [1, 2]


def test_hip_runtime_info_drm_stride() -> None:
    impl = gpuq.mock(
        hip_count=3,
        cuda_count=None,
        hip_drm=128,
        hip_node_idx=2,
        hip_pids=[],
        _hip_drm_stride=8,
    )
    with impl:
        g0 = gpuq.get(0, visible_only=False)
        g1 = gpuq.get(1, visible_only=False)
        g2 = gpuq.get(2, visible_only=False)
        assert g0.hip_info.drm == 128
        assert g1.hip_info.drm == 136
        assert g2.hip_info.drm == 144
        assert g0.hip_info.node_idx == 2
        assert g1.hip_info.node_idx == 3
        assert g2.hip_info.node_idx == 4

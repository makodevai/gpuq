import os
from typing import Generator, Callable
from unittest.mock import patch
from contextlib import contextmanager, ExitStack

import gpuinfo.hip


_pid_gpus_output = b"""\


============================ ROCm System Management Interface ============================
================================== GPUs Indexed by PID ===================================
PID 78912 is using 0 DRM device(s)
PID 79301 is using 1 DRM device(s):
1 
PID 9710 is using 0 DRM device(s)
PID 1951076 is using 0 DRM device(s)
PID 1949829 is using 2 DRM device(s):
0 
1 
PID 10678 is using 1 DRM device(s):
0 
==========================================================================================
================================== End of ROCm SMI Log ===================================
"""


@contextmanager
def mock_hip_tree() -> Generator[None, None, None]:
    hip_tree = [
        {"gfx": "942", "drm": 128, "node": 2},
    ]

    with ExitStack() as stack:
        stack.enter_context(
            patch("gpuinfo.hip._get_hip_nodes_info", return_value=hip_tree)
        )
        stack.enter_context(
            patch("subprocess.check_output", return_value=_pid_gpus_output)
        )
        yield


def get_mocked_read_file() -> Callable[[str], str]:
    def mocked_read_file(file: str) -> str:
        file = str(file)
        assert file.startswith("/sys/class/kfd/kfd/topology/nodes/")
        node_idx = int(file.split(os.path.sep)[7])
        if node_idx < 0 or node_idx > 9:
            raise FileNotFoundError()
        if node_idx < 2:
            return """\
cpu_cores_count 96
simd_count 0
mem_banks_count 1
caches_count 0
io_links_count 5
p2p_links_count 3
cpu_core_id_base 0
simd_id_base 0
max_waves_per_simd 0
lds_size_in_kb 0
gds_size_in_kb 0
num_gws 0
wave_front_size 0
array_count 0
simd_arrays_per_engine 0
cu_per_simd_array 0
simd_per_cu 0
max_slots_scratch_cu 0
gfx_target_version 0
vendor_id 0
device_id 0
location_id 0
domain 0
drm_render_minor 0
hive_id 0
num_sdma_engines 0
num_sdma_xgmi_engines 0
num_sdma_queues_per_engine 0
num_cp_queues 0
max_engine_clk_ccompute 2400
"""

        return """\
cpu_cores_count 0
simd_count 1216
mem_banks_count 1
caches_count 626
io_links_count 7
p2p_links_count 1
cpu_core_id_base 0
simd_id_base 2147487784
max_waves_per_simd 8
lds_size_in_kb 64
gds_size_in_kb 0
num_gws 64
wave_front_size 64
array_count 32
simd_arrays_per_engine 1
cu_per_simd_array 10
simd_per_cu 4
max_slots_scratch_cu 32
gfx_target_version 90402
vendor_id 4098
device_id 29857
location_id 18176
domain 0
drm_render_minor {}
hive_id 10631624818852212319
num_sdma_engines 2
num_sdma_xgmi_engines 14
num_sdma_queues_per_engine 8
num_cp_queues 24
max_engine_clk_fcompute 2100
local_mem_size 0
fw_version 166
capability 746037888
debug_prop 1511
sdma_fw_version 22
unique_id 4376936076086227261
num_xcc 8
max_engine_clk_ccompute 2400
""".format(
            128 + (node_idx - 2) * 8
        )

    return mocked_read_file


@contextmanager
def mock_fs() -> Generator[None, None, None]:
    with ExitStack() as stack:
        stack.enter_context(patch("os.path.exists", return_value=True))
        stack.enter_context(patch("os.listdir", return_value=["1", "2", "0"]))
        stack.enter_context(
            patch("gpuinfo.hip._read_file", new_callable=get_mocked_read_file)
        )
        stack.enter_context(
            patch("subprocess.check_output", return_value=_pid_gpus_output)
        )
        yield


def test_get_hip_info_1() -> None:
    with mock_hip_tree():
        data = gpuinfo.hip.get_hip_info(0)
        assert data is not None
        assert data.index == 0
        assert data.drm == 128
        assert data.gfx == "942"
        assert data.node_idx == 2
        assert data.pids == [1949829, 10678]


def test_get_hip_info_failure() -> None:
    with mock_hip_tree():
        data = gpuinfo.hip.get_hip_info(1)
        assert data is None


def test_get_hip_info_fs() -> None:
    with mock_fs():
        data = gpuinfo.hip.get_hip_info(0)
        assert data is not None
        assert data.index == 0
        assert data.drm == 128
        assert data.gfx == "942"
        assert data.node_idx == 2
        assert data.pids == [1949829, 10678]

        data2 = gpuinfo.hip.get_hip_info(1)
        assert data2 is None


def test_hip_info_unordered_listdir() -> None:
    with mock_fs():
        with patch(
            "os.listdir",
            return_value=["7", "5", "3", "1", "8", "6", "4", "2", "0", "9"],
        ):
            data = gpuinfo.hip.get_hip_info(0)
            assert data is not None
            assert data.index == 0
            assert data.node_idx == 2

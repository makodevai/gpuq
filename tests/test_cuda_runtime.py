import pytest
from unittest.mock import Mock, patch
from contextlib import contextmanager, ExitStack

import gpuinfo.cuda



@contextmanager
def mock_get_gpu_status():
    with ExitStack() as stack:
        stack.enter_context(patch("gpuinfo.cuda._get_num_gpus", return_value=1))
        stack.enter_context(patch("gpuinfo.cuda.get_gpu_status", return_value={ "utilisation": 11, "used_memory": 1552, "pids": [1, 1024] }))
        yield


def get_mocked_nvidia_smi():
    def mocked_nvidia_smi(args):
        if '-L' in args:
            return b'''\
GPU 0: NVIDIA GeForce Mako (UUID: GPU-cae6066b-8667-ce45-a986-d93f33d8573f)
'''

        return b'''\
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.85.02    Driver Version: 510.85.02    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 Off |                  N/A |
| N/A   32C    P8    N/A /  N/A |   3546MiB /  4096MiB |     81%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2806      G   /usr/lib/xorg/Xorg                296MiB |
|    0   N/A  N/A     28690      C   ...abcdef/venv/bin/python3.7     3069MiB |
+-----------------------------------------------------------------------------+
'''
    return mocked_nvidia_smi


@contextmanager
def mock_nvidia_smi():
    with ExitStack() as stack:
        stack.enter_context(patch("gpuinfo.cuda._get_nvidia_smi_path", return_value="mock"))
        stack.enter_context(patch("subprocess.check_output", new_callable=get_mocked_nvidia_smi))
        yield


def test_get_gpu_status_mock():
    with mock_get_gpu_status():
        data = gpuinfo.cuda.get_gpu_status(0)
        assert data["utilisation"] == 11
        assert data["used_memory"] == 1552
        assert data["pids"] == [1, 1024]


def test_get_cuda_info_1():
    with mock_get_gpu_status():
        data = gpuinfo.cuda.get_cuda_info(0)
        assert data.index == 0
        assert data.utilisation == 11
        assert data.used_memory == 1552
        assert data.pids == [1, 1024]


def test_get_cuda_info_failing():
    with mock_get_gpu_status():
        data = gpuinfo.cuda.get_cuda_info(1)
        assert data is None


def test_get_cuda_info_nvidia_smi():
    with mock_nvidia_smi():
        data = gpuinfo.cuda.get_cuda_info(0)
        assert data.utilisation == 81
        assert data.used_memory == 3546
        assert data.pids == [2806, 28690]

        data2 = gpuinfo.cuda.get_cuda_info(1)
        assert data2 is None

import os
import re
import subprocess
from pathlib import Path
from functools import cache
from dataclasses import dataclass


_amd_nodes_tree = "/sys/class/kfd/kfd/topology/nodes/"
_amd_node_info_file = "properties"
_amd_node_gfx_ver = re.compile(r"gfx_target_version ([0-9a-f]+)")
_amd_node_drm_minor = re.compile(r"drm_render_minor ([0-9]+)")
_amd_pid_gpus = re.compile(r"PID ([0-9]+) is using ([0-9]+) DRM device\(s\):\n(([0-9]+\s*)+)", re.MULTILINE)


@dataclass
class HipRuntimeInfo:
    index: int
    gfx: str
    drm: int
    node_idx: int

    @property
    def pids(self) -> list[int]:
        return get_gpu_pids(self.index)


@dataclass
class HipRuntimeInfoMock(HipRuntimeInfo):
    def __init__(self, index: int, gfx: str, drm: int, node_idx: int, pids: list[int]):
        super().__init__(index, gfx, drm, node_idx)
        self.__pids = pids

    @property
    def pids(self) -> list[int]:
        return self.__pids


@cache
def _get_hip_nodes_info() -> list[dict[str, str | int]]:
    if os.path.exists(_amd_nodes_tree):
        gpus: list[dict[str, str | int]] = []
        for node in os.listdir(_amd_nodes_tree):
            node_idx = int(node)
            node = os.path.join(_amd_nodes_tree, node, _amd_node_info_file)
            try:
                info = Path(node).read_text()
            except:
                continue
            gfx_match = _amd_node_gfx_ver.search(info)
            drm_match = _amd_node_drm_minor.search(info)
            if (
                gfx_match
                and drm_match
                and gfx_match.group(1)
                and gfx_match.group(1) != "0"
            ):

                def parse_gfx(gfx: str) -> str:
                    major, minor, rev = gfx[:-4], gfx[-4:-2], gfx[-2:]
                    major, minor, rev = major, hex(int(minor)), hex(int(rev))
                    major, minor, rev = major, minor[2:], rev[2:]
                    return major + minor + rev

                parsed_gfx = parse_gfx(gfx_match.group(1) or "")
                parsed_drm = drm_match.group(1)
                gpus.append(
                    {"gfx": parsed_gfx, "drm": int(parsed_drm), "node": int(node_idx)}
                )

        return gpus

    return []


def get_gpu_pids(gpu_idx: int) -> list[int]:
    try:
        output = subprocess.check_output(["rocm-smi", "--showpidgpus"]).decode('UTF-8')
    except:
        return []

    ret = []
    match: re.Match[str]
    for match in _amd_pid_gpus.finditer(output):
        pid = int(match.group(1))
        num_gpus = int(match.group(2))
        if num_gpus <= 0:
            continue

        gpus = { int(gpu) for gpu in match.group(3).splitlines() if gpu }
        if gpu_idx in gpus:
            ret.append(pid)

    return ret


def get_hip_info(gpu_idx: int) -> HipRuntimeInfo | None:
    if gpu_idx < 0:
        return None

    nodes_info = _get_hip_nodes_info()
    if gpu_idx >= len(nodes_info):
        return None
    
    return HipRuntimeInfo(
        index=gpu_idx,
        gfx=nodes_info[gpu_idx]["gfx"],
        drm=nodes_info[gpu_idx]["drm"],
        node_idx=nodes_info[gpu_idx]["node"],
    )

from typing import Any


version: str = "1.5.2"
repo: str = "unknown"
commit: str = "unknown"
has_repo: bool = False


try:
    import git
    from pathlib import Path

    try:
        r = git.Repo(Path(__file__).parents[1])
        has_repo = True

        if not r.remotes:
            repo = "local"
        else:
            repo = str(r.remotes.origin.url)

        commit = str(r.head.commit.hexsha)
        status = []
        if r.is_dirty():
            status.append("dirty")
        if r.untracked_files:
            status.append(f"+{len(r.untracked_files)} untracked")
        if status:
            commit += f' ({",".join(status)})'
    except git.InvalidGitRepositoryError:
        raise ImportError()
except ImportError:
    pass

try:
    import importlib.util
    from pathlib import Path

    _dist_info_file = Path(__file__).parent.joinpath("_dist_info.py")
    if _dist_info_file.exists():
        _spec = importlib.util.spec_from_file_location("_dist_info", _dist_info_file)
        assert _spec is not None
        _dist_info = importlib.util.module_from_spec(_spec)
        assert _dist_info is not None
        assert _spec.loader is not None
        _spec.loader.exec_module(_dist_info)
        assert not has_repo, "_dist_info should not exist when repo is in place"
        assert version == _dist_info.version
        repo = str(_dist_info.repo)
        commit = str(_dist_info.commit)
except (ImportError, SystemError):
    pass


def info() -> dict[str, Any]:
    g = globals()
    return {k: g[k] for k in __all__}


__all__ = ["version", "repo", "commit", "has_repo"]

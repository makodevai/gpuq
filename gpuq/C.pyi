class Properties:
    ord: int
    uuid: str
    provider: str
    index: int
    name: str
    major: int
    minor: int
    total_memory: int
    sms_count: int
    sm_threads: int
    sm_shared_memory: int
    sm_blocks: int
    block_threads: int
    block_shared_memory: int
    warp_size: int
    l2_cache_size: int
    concurrent_threads: bool
    async_engines_count: int
    cooperative: bool

def checkcuda() -> str: ...
def checkamd() -> str: ...
def count() -> int: ...
def get(index: int) -> Properties: ...

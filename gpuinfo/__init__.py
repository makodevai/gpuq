from .utils import add_module_properties, staticproperty


from . import C


class Properties():
    def __init__(self, cobj):
        self.cobj = cobj

    @property
    def provider(self):
        return self.cobj.provider

    @property
    def index(self):
        return self.cobj.index

    @property
    def name(self):
        return self.cobj.name

    @property
    def major(self):
        return self.cobj.major

    @property
    def minor(self):
        return self.cobj.minor

    @property
    def total_memory(self):
        return self.cobj.total_memory

    @property
    def sms_count(self):
        return self.cobj.sms_count

    @property
    def sm_threads(self):
        return self.cobj.sm_threads

    @property
    def sm_shared_memory(self):
        return self.cobj.sm_shared_memory

    @property
    def sm_registers(self):
        return self.cobj.sm_registers

    @property
    def sm_blocks(self):
        return self.cobj.sm_blocks

    @property
    def block_threads(self):
        return self.cobj.block_threads

    @property
    def block_shared_memory(self):
        return self.cobj.block_shared_memory

    @property
    def block_registers(self):
        return self.cobj.block_registers

    @property
    def warp_size(self):
        return self.cobj.warp_size

    @property
    def l2_cache_size(self):
        return self.cobj.l2_cache_size

    @property
    def concurrent_kernels(self):
        return self.cobj.concurrent_kernels

    @property
    def async_engines_count(self):
        return self.cobj.async_engines_count

    @property
    def cooperative(self):
        return self.cobj.cooperative

    def asdict(self):
        return {
            'provider': self.provider,
            'index': self.index,
            'name': self.name,
            'major': self.major,
            'minor': self.minor,
            'total_memory': self.total_memory,
            'sms_count': self.sms_count,
            'sm_threads': self.sm_threads,
            'sm_shared_memory': self.sm_shared_memory,
            'sm_registers': self.sm_registers,
            'sm_blocks': self.sm_blocks,
            'block_threads': self.block_threads,
            'block_shared_memory': self.block_shared_memory,
            'block_registers': self.block_registers,
            'warp_size': self.warp_size,
            'l2_cache_size': self.l2_cache_size,
            'concurrent_kernels': self.concurrent_kernels,
            'async_engines_count': self.async_engines_count,
            'cooperative': self.cooperative
        }

    def __str__(self):
        props = self.asdict()
        provider = props.pop('provider')
        index = props.pop('index')
        return f'{type(self).__module__}.{type(self).__qualname__}(provider={provider!r}, idx={index}):\n    ' + '\n    '.join(f'{key}: {value}' for key, value in props.items()) + '\n'


def count():
    return C.count()


def get(idx):
    ret = C.get(idx)
    return Properties(ret)


def _get_version():
    from . import version
    return version.version

def _get_has_repo():
    from . import version
    return version.has_repo

def _get_repo():
    from . import version
    return version.repo

def _get_commit():
    from . import version
    return version.commit


add_module_properties(__name__, {
    '__version__': staticproperty(staticmethod(_get_version)),
    '__has_repo__': staticproperty(staticmethod(_get_has_repo)),
    '__repo__': staticproperty(staticmethod(_get_repo)),
    '__commit__': staticproperty(staticmethod(_get_commit))
})

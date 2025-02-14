from enum import IntFlag, auto

from .utils import add_module_properties, staticproperty


from . import C


class Provider(IntFlag):
    ANY = 0

    CUDA = auto()
    HIP = auto()

    ALL = CUDA|HIP


_provider_query_map = {
    Provider.CUDA: C.hascuda,
    Provider.HIP: C.hasamd
}


class Properties():
    def __init__(self, cobj):
        self.cobj = cobj

    @property
    def provider(self):
        return Provider[self.cobj.provider]

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


def query(provider: Provider = Provider.ANY, required: Provider = None) -> list[Properties]:
    ''' Return a list of all GPUs matching the provided criteria.

        ``provider`` should be a bitwise-or'ed mask of providers whose GPUs should
        be returned. The values of ``ALL``, ``ANY`` and ``None`` all mean that
        all providers should be included when returning GPUs.

        ``required`` is another bitwise-or'ed mask of providers that can additionally
        be used to make the function raise an error (RuntimeError) if GPUs of
        a particular provider are not present:
             - ``None`` means nothing is required
             - ``True`` means that at least one GPU should be returned
             - ``ANY`` means at least one GPU should be present (but not necessarily returned,
                see the note below)
             - `anything else (including ``ALL``) means that at least one GPU of each provider
                included in the mask has to be present.

        > **Note:** ```required`` and ``provider`` are mostly independent. For example,
        > a call like ``query(provider=CUDA, required=HIP)`` is valid and will raise an
        > error if there are no HIP devices but will only return CUDA devices (potentially
        > an empty list). This means, ``required=ANY`` might be a bit counter-intuitive,
        > since it will only fail if there are no GPUs whatsoever on the system.
        > The only exception to this rule is the ``required=True`` case, which
        > could be understood as "make sure at least one GPU is returned", while
        > taking into account the provided ``providers`` value.
    '''
    nonempty = False
    if required is True:
        required = Provider.ANY
        nonempty = True

    if provider == Provider.ANY or provider is None:
        provider = Provider.ALL

    if required:
        for p in Provider:
            if p & required:
                if not _provider_query_map[p]():
                    raise RuntimeError(f'Provider {p.name} is required but the relevant runtime is missing from the system!')

    num = C.count()

    if not num:
        if required is not None or nonempty:
            raise RuntimeError('No GPUs detected')
        return []

    ret = []

    for idx in range(num):
        dev = C.get(idx)
        prov = Provider[dev.provider]
        if required is not None and prov & required:
            required &= ~prov # mark the current provider as no longer required

        if provider & prov:
            ret.append(Properties(dev))

    if required:
        missing = [p for p in Provider if p & required]
        raise RuntimeError(f'GPUs of the following required providers could not be found: {missing}')

    if not ret and nonempty:
        raise RuntimeError('No suitable GPUs detected')

    return ret


def count(provider: Provider = Provider.ALL):
    ''' Return the overall amount of GPUs for the specified provider (by default all providers).

        ``providers`` can be a bitwise mask of valid providers.
    '''
    if provider == Provider.ANY or provider is None:
        provider = Provider.ALL

    if provider == Provider.ALL:
        return C.count()
    else:
        return len(query(provider=provider, required=None))


def get(idx, provider: Provider = Provider.ALL):
    ''' Return the ``idx``-th GPU from the list of GPus for the specified provider(s).
    '''
    if provider == Provider.ANY or provider is None:
        provider = Provider.ALL

    if provider == Provider.ALL:
        ret = C.get(idx)
        return Properties(ret)
    else:
        ret = query(provider=provider, required=None)
        if not ret:
            raise RuntimeError('No GPUs available')
        if idx < 0 or idx >= len(ret):
            raise ValueError('Invalid GPU index')
        return ret[idx]


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

import os
from contextlib import contextmanager
from enum import IntFlag, auto

from .utils import add_module_properties, staticproperty


from . import C


class Provider(IntFlag):
    ANY = 0

    CUDA = auto()
    HIP = auto()

    ALL = CUDA|HIP


@contextmanager
def _save_visible(clear=True):
    cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    hip = os.environ.get('HIP_VISIBLE_DEVICES', None)

    def _is_int(value, _prefix):
        try:
            value = int(value)
            return True
        except:
            raise ValueError(f'{_prefix}_VISIBLE_DEVICES environment variable contains values that are not integer - this is currently not supported: {value!r}') from None

    if cuda is not None:
        parsed_cuda = { int(g) for g in cuda.split(',') if _is_int(g, 'CUDA') }
        parsed_cuda = sorted(list(parsed_cuda))
    else:
        parsed_cuda = None

    if hip is not None:
        parsed_hip = { int(g) for g in hip.split(',') if _is_int(g, 'HIP') }
        parsed_hip = sorted(list(parsed_hip))
    else:
        parsed_hip = parsed_cuda

    if clear:
        if cuda is not None:
            del os.environ['CUDA_VISIBLE_DEVICES']
        if hip is not None:
            del os.environ['HIP_VISIBLE_DEVICES']

    try:
        yield { Provider.CUDA: parsed_cuda, Provider.HIP: parsed_hip }
    finally:
        if clear:
            if cuda is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda
            if hip is not None:
                os.environ['HIP_VISIBLE_DEVICES'] = hip


def _global_to_visible(system_index: int, visible: list[int]):
    if visible is None:
        return system_index

    try:
        return visible.index(system_index)
    except ValueError:
        return None


_provider_query_map = {
    Provider.CUDA: C.hascuda,
    Provider.HIP: C.hasamd
}


class Properties():
    def __init__(self, cobj, local_index):
        self.cobj = cobj
        self.local_index = local_index

    @property
    def ord(self):
        return self.cobj.ord

    @property
    def provider(self):
        return Provider[self.cobj.provider]

    @property
    def index(self):
        return self.local_index

    @property
    def system_index(self):
        return self.cobj.index

    @property
    def is_visible(self):
        return self.index is not None

    @property
    def name(self):
        return self.cobj.name

    @property
    def short_name(self):
        from .short_names import short_names
        return short_names.get(self.cobj.name, None)

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

    def asdict(self, strip_index=False):
        ret = {
            'ord': self.ord,
            'provider': self.provider,
            'index': self.index,
            'system_index': self.system_index,
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

        if strip_index:
            del ret['ord']
            del ret['index']
            del ret['system_index']

        return ret

    def __eq__(self, other):
        ''' Return True if self and other are equivalent GPUs.

            If you want to check if two GPus are the same physical devices,
            simply compare their indices.
        '''
        if not isinstance(other, Properties):
            return False
        if self.index == other.index:
            return True
        return self.asdict(strip_index=True) == other.asdict(strip_index=True)

    def __str__(self):
        props = self.asdict(strip_index=True)
        del props['provider']
        del props['name']
        return self.__repr__() + '{\n    ' + '\n    '.join(f'{key}: {value}' for key, value in props.items()) + '\n}'

    def __repr__(self):
        return f'{type(self).__module__}.{type(self).__qualname__}({self.provider.name}[{self.system_index} -> {self.index}], {self.name!r})'


def query(provider: Provider = Provider.ANY, required: Provider = None, visible_only: bool = True) -> list[Properties]:
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

        If ``visible_only`` is True, any processing of GPU by function (including checking
        for providers and GPUs as described above) will only consider GPUs that are visible
        according to the relevant *_VISIBLE_DEVICES environmental variable. Otherwise
        the variables are ignored and all GPUs are always considered.
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

    with _save_visible() as visible:
        num = C.count()

        if not num:
            if required is not None or nonempty:
                raise RuntimeError('No GPUs detected')
            return []

        ret = []

        for idx in range(num):
            dev = C.get(idx)
            prov = Provider[dev.provider]

            visible_set = visible.get(prov)
            local_index = _global_to_visible(dev.index, visible_set)
            if visible_only and local_index is None: # not visible
                continue

            if required is not None and prov & required:
                required &= ~prov # mark the current provider as no longer required

            if provider & prov:
                ret.append(Properties(dev, local_index))

        if required:
            missing = [p for p in Provider if p & required]
            raise RuntimeError(f'GPUs of the following required providers could not be found: {missing}')

        if not ret and nonempty:
            raise RuntimeError('No suitable GPUs detected')

        return ret


def count(provider: Provider = Provider.ALL, visible_only: bool = False):
    ''' Return the overall amount of GPUs for the specified provider (by default all providers).

        ``providers`` can be a bitwise mask of valid providers.
        if ``visible_only`` is True, return the number of matching GPUs that visible according to
        *_VISIBLE_DEVICES environment variables. Otherwise the number of all GPUs matching the
        criteria is returned.
    '''
    if provider == Provider.ANY or provider is None:
        provider = Provider.ALL

    if provider == Provider.ALL:
        if visible_only:
            return C.count()
        else:
            with _save_visible():
                return C.count()
    else:
        return len(query(provider=provider, required=None, visible_only=visible_only))


def get(idx, provider: Provider = Provider.ALL, visible_only: bool = False):
    ''' Return the ``idx``-th GPU from the list of GPus for the specified provider(s).
        If ``visible_only`` is True, only visible devices according to *_VISIBLE_DEVICES
        environment variables are considered for indexing (see ``count``).
    '''
    if provider == Provider.ANY or provider is None:
        provider = Provider.ALL

    if provider == Provider.ALL and not visible_only:
        with _save_visible() as visible:
            ret = C.get(idx)
            prov = Provider[ret.provider]
            visible_set = visible,get(prov)
            local_index = _global_to_visible(ret.index, visible_set)
            return Properties(ret, local_index)
    else:
        ret = query(provider=provider, required=None, visible_only=visible_only)
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

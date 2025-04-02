
class Properties():
    def __init__(self, cobj, local_index):
        self.cobj = cobj
        self.local_index = local_index

    @property
    def ord(self) -> int:
        ''' Ordinal of the GPU, across all providers and devices. Specific to the gpuinfo package.
        '''
        return self.cobj.ord

    @property
    def provider(self) -> str:
        ''' Which runtime provides the GPU. Currently CUDA or HIP
        '''
        return Provider[self.cobj.provider]

    @property
    def index(self) -> (int | None):
        ''' Index of the GPU as seen by the calling process. This can be affected by
            various *_VISIBLE_DEVICES environment variables.

            Can be ``None`` if the GPU is not visible by this process. Also see ``is_visible``
            for more information.

            This index is provider-specific.

            > **Note:** visibility is determined at the moment of constructing the object and will not
            > reflect any changes made later.
        '''
        return self.local_index

    @property
    def system_index(self) -> int:
        ''' System-wide index of the GPU, i.e., its index when ignoring *_VISIBLE_DEVICES.

            This index is provider-specific.

            > **Note:** system-wide index is determined by temporarily removing *_VISIBLE_DEVICES
            > variables.
            > This might cause race conditions if the variables are also used/modified by other
            > parts of the system at the same time. Please keep this in mind when using the package.
        '''
        return self.cobj.index

    @property
    def is_visible(self) -> bool:
        ''' Whether the GPU is visible by the current process, per the relevant *_VISIBLE_DEVICES
            environment variables.

            > **Note:** visibility is determined at the moment of constructing the object and will not
            > reflect any changes made later.

            > **Note:** the implementation will temporarily remove any *_VISIBLE_DEVICES variables
            > when obtaining information about the GPU, to correctly report other properties.
            > This might cause race conditions if the variables are also used/modified by other
            > parts of the system at the same time. Please keep this in mind when using the package.
        '''
        return self.index is not None

    @property
    def name(self) -> str:
        ''' Name of the GPU
        '''
        return self.cobj.name

    @property
    def short_name(self) -> (str | None):
        ''' Short name of the GPU, if known. Otherwise None.
        '''
        from .short_names import short_names
        return short_names.get(self.cobj.name, None)

    @property
    def major(self) -> int:
        return self.cobj.major

    @property
    def minor(self) -> int:
        return self.cobj.minor

    @property
    def total_memory(self) -> int:
        return self.cobj.total_memory

    @property
    def sms_count(self) -> int:
        return self.cobj.sms_count

    @property
    def sm_threads(self) -> int:
        return self.cobj.sm_threads

    @property
    def sm_shared_memory(self) -> int:
        return self.cobj.sm_shared_memory

    @property
    def sm_registers(self) -> int:
        return self.cobj.sm_registers

    @property
    def sm_blocks(self) -> int:
        return self.cobj.sm_blocks

    @property
    def block_threads(self) -> int:
        return self.cobj.block_threads

    @property
    def block_shared_memory(self) -> int:
        return self.cobj.block_shared_memory

    @property
    def block_registers(self) -> int:
        return self.cobj.block_registers

    @property
    def warp_size(self) -> int:
        return self.cobj.warp_size

    @property
    def l2_cache_size(self) -> int:
        return self.cobj.l2_cache_size

    @property
    def concurrent_kernels(self) -> bool:
        return self.cobj.concurrent_kernels

    @property
    def async_engines_count(self) -> int:
        return self.cobj.async_engines_count

    @property
    def cooperative(self) -> bool:
        return self.cobj.cooperative

    def asdict(self, strip_index=False):
        ret = {
            'ord': self.ord,
            'provider': self.provider.name,
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

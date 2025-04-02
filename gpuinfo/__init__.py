from threading import local
from contextlib import contextmanager
from typing import Generator, Literal

from .datatypes import Provider, Properties
from .impl import Implementation, GenuineImplementation, MockImplementation
from .utils import add_module_properties, staticproperty


_current_implementation = local()
_default_impl = GenuineImplementation()


def _get_impl() -> Implementation:
    return getattr(_current_implementation, "value", _default_impl)


def _set_impl(impl: Implementation | None) -> Implementation:
    current = _get_impl()
    _current_implementation.value = impl if impl is not None else _default_impl
    return current


@contextmanager
def _with_impl(impl: Implementation | None) -> Generator[Implementation, None, None]:
    if impl is None:
        impl = _default_impl
    curr = _set_impl(impl)
    try:
        yield impl
    finally:
        _set_impl(curr)


def _global_to_visible(system_index: int, visible: list[int] | None):
    if visible is None:
        return system_index

    try:
        return visible.index(system_index)
    except ValueError:
        return None


def query(
    provider: Provider = Provider.ANY,
    required: Provider | None | Literal[True] = None,
    visible_only: bool = True,
    impl: Implementation | None = None,
) -> list[Properties]:
    """Return a list of all GPUs matching the provided criteria.

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

    > **Note:** the implementation will temporarily remove any *_VISIBLE_DEVICES variables
    > when obtaining information about GPUs, regardless of ``visible_only`` argument.
    > This might cause race conditions if the variables are also used/modified by other
    > parts of the system at the same time. Please keep this in mind when using it.
    """
    nonempty = False
    if required is True:
        required = Provider.ANY
        nonempty = True

    if provider == Provider.ANY or provider is None:
        provider = Provider.ALL

    if impl is None:
        impl = _get_impl()

    if required:
        for p in Provider:
            if p & required:
                if not impl.provider_check(p):
                    raise RuntimeError(
                        f"Provider {p.name} is required but the relevant runtime is missing from the system!"
                    )

    with impl.save_visible() as visible:
        num = impl.c_count()

        if not num:
            if required is not None or nonempty:
                raise RuntimeError("No GPUs detected")
            return []

        ret = []

        for idx in range(num):
            dev = impl.c_get(idx)
            prov = Provider[dev.provider]

            visible_set = visible.get(prov)
            local_index = _global_to_visible(dev.index, visible_set)
            if visible_only and local_index is None:  # not visible
                continue

            if required is not None and prov & required:
                required &= ~prov  # mark the current provider as no longer required

            if provider & prov:
                ret.append(Properties(dev, local_index, impl))

        if required:
            missing = [p for p in Provider if p & required]
            raise RuntimeError(
                f"GPUs of the following required providers could not be found: {missing}"
            )

        if not ret and nonempty:
            raise RuntimeError("No suitable GPUs detected")

        return ret


def count(
    provider: Provider = Provider.ALL,
    visible_only: bool = False,
    impl: Implementation | None = None,
) -> int:
    """Return the overall amount of GPUs for the specified provider (by default all providers).

    ``providers`` can be a bitwise mask of valid providers.
    if ``visible_only`` is True, return the number of matching GPUs that visible according to
    *_VISIBLE_DEVICES environment variables. Otherwise the number of all GPUs matching the
    criteria is returned.

    > **Note:** the implementation will temporarily remove any *_VISIBLE_DEVICES variables
    > when obtaining information about GPUs, if ``visible_only`` is False.
    > This might cause race conditions if the variables are also used/modified by other
    > parts of the system at the same time. Please keep this in mind when using it.
    """
    if provider == Provider.ANY or provider is None:
        provider = Provider.ALL

    if impl is None:
        impl = _get_impl()

    if provider == Provider.ALL:
        if visible_only:
            return impl.c_count()
        else:
            with impl.save_visible():
                return impl.c_count()
    else:
        return len(
            query(
                provider=provider, required=None, visible_only=visible_only, impl=impl
            )
        )


def get(
    idx,
    provider: Provider = Provider.ALL,
    visible_only: bool = False,
    impl: Implementation | None = None,
) -> Properties:
    """Return the ``idx``-th GPU from the list of GPus for the specified provider(s).
    If ``visible_only`` is True, only visible devices according to *_VISIBLE_DEVICES
    environment variables are considered for indexing (see ``count``).

    > **Note:** the implementation will temporarily remove any *_VISIBLE_DEVICES variables
    > when obtaining information about GPUs, regardless of ``visible_only`` argument.
    > This might cause race conditions if the variables are also used/modified by other
    > parts of the system at the same time. Please keep this in mind when using it.
    """
    if provider == Provider.ANY or provider is None:
        provider = Provider.ALL

    if impl is None:
        impl = _get_impl()

    if provider == Provider.ALL and not visible_only:
        with impl.save_visible() as visible:
            ret = impl.c_get(idx)
            prov = Provider[ret.provider]
            visible_set = visible.get(prov)
            local_index = _global_to_visible(ret.index, visible_set)
            return Properties(ret, local_index, impl)
    else:
        ret = query(
            provider=provider, required=None, visible_only=visible_only, impl=impl
        )
        if not ret:
            raise RuntimeError("No GPUs available")
        if idx < 0 or idx >= len(ret):
            raise IndexError("Invalid GPU index")
        return ret[idx]


def hasprovider(p: Provider, impl: Implementation | None = None) -> bool:
    if impl is None:
        impl = _get_impl()
    return impl.provider_check(p)


def hascuda(impl: Implementation | None = None) -> bool:
    if impl is None:
        impl = _get_impl()
    return impl.provider_check(Provider.CUDA)


def hasamd(impl: Implementation | None = None) -> bool:
    if impl is None:
        impl = _get_impl()
    return impl.provider_check(Provider.HIP)


def mock(
    cuda_count: int | None = 1,
    hip_count: int | None = None,
    cuda_visible: list[int] | None = None,
    hip_visible: list[int] | None = None,
    name: str = "MockDevice",
    major: int = 1,
    minor: int = 2,
    total_memory: int = 8 * 1024**3,
    sms_count: int = 12,
    sm_threads: int = 2048,
    sm_shared_memory: int = 16 * 1024,
    sm_registers: int = 512,
    sm_blocks: int = 4,
    block_threads: int = 1024,
    block_shared_memory: int = 8 * 1024,
    block_registers: int = 256,
    warp_size: int = 32,
    l2_cache_size: int = 8 * 1024**2,
    concurrent_kernels: bool = True,
    async_engines_count: int = 0,
    cooperative: bool = True,
) -> Implementation:
    return MockImplementation(**locals())


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


add_module_properties(
    __name__,
    {
        "__version__": staticproperty(staticmethod(_get_version)),
        "__has_repo__": staticproperty(staticmethod(_get_has_repo)),
        "__repo__": staticproperty(staticmethod(_get_repo)),
        "__commit__": staticproperty(staticmethod(_get_commit)),
    },
)

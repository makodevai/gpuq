# *gpuq* - multi-vendor *GPU* *q*uerying utility with minimal dependencies

This small library is a direct answer to the lack of a lightweight, cross-compatible utility to query available GPUs - regardless of what vendor, distro, or overall environment one might be using.

In particular, the implementation meets the following requirements:
 - works with multiple downstream runtimes (currently supported: CUDA and HIP)
    - will also work if you have multiple present at the same time (do you really, though?)
 - no build- or install-time dependencies (including any python packages)
 - any runtime dependencies are soft - unless the user explicitly asks for the status/presence of a particular downstream runtime, most methods will fail silently
 - consequently, the package should install and run on pretty much any machine
    - your laptop does not have a GPU? -> the package will report 0 GPUs available (duh), no exceptions, linker errors, etc.
 - allows for easy mocking (for unit tests, etc.)
 - fully typed (conforms to `mypy --strict` checking)

Compared to some existing alternatives, it has the following differences:
 - `torch.cuda` - not lightweight, also requires a different wheel for NVidia and HIP
 - `gputil` - NVidia specific, also broken dependencies (as of 2025)
 - `gpuinfo` - NVidia specific, broken `import gpuinfo`...
 - `gpuinfonv` - NVidia specific, requires pynvml
 - `pyamdgpuinfo` - AMD specific
 - `igpu` - NVidia specific, broken installation (as of 2025)
 - and so on...

The primary functionality offered is:
 - check how many GPUs are available
 - query properties for each available device - will tell you some basic info about the provider (CUDA/HIP) and other info similar to `cudaGetDeviceProperties`
    - the returned list is not comprehensive, though
 - respects `*_VISIBLE_DEVICES` and provides mapping between local (visible) and global indices
    - **NOTE: this temporarily modifies env variables and therefore is not thread-safe**
 - if requested, lazily provides some runtime information about each GPU as well
    - in particular, PIDs of processes using the GPU will be returned
    - NOTE: this is currently done rather naively by parsing outputs of tools like `nvidia-smi` or `rocm-smi`
 - allows to check for runtime errors that might have occurred while trying to load 

### How it works:

The implementation will attempt to dynamically lazy-load `libcudart.so` and `libamdhip64.so` at runtime.
For GPUs to be properly reported, the libraries have to be found by the dynamic linker at the moment any relevant function call is made for the first time.
(If a library fails to load, loading will be retried every time a function call is made).

## Examples

Return the number of available GPUs:
```python
import gpuq as G

print(G.count()) # this includes GPUs from all providers, disregarding *_VISIBLE_DEVICES
print(G.count(visible_only=True)) # only visible GPUs from all providers
print(G.count(provider=G.Provider.HIP, visible_only=True)) # only visible HIP devices
# etc.
```

Return a list of gpu properties:
```python
import gpuq as G

for gpu in G.query(visible_only=True, provider=G.Provider.ANY):
    print(gpu.name)

# return all visible GPUs, raise an error if no CUDA devices are present
# (note: the required check if done against the global set, not the return set - see the docs)
gpus = G.query(visible_only=True, provider=G.Provider.ANY, required=G.Provider.CUDA)
```

Provide mapping between local and global GPU indices:
```python
import gpuq as G

# assume a system with 8 GPUs and CUDA_VISIBLE_DEVICES=1,7
for gpu in G.query(): # by default return visible GPUs only
    print(gpu.index, gpu.system_index)

# should print:
# 0 1
# 1 7

for gpu in G.query(visible_only=False):
    print(gpu.index, gpu.system_index, gpu.is_visible)

# should print:
# None 0 False
# 0 1 True
# None 2 False
# None 3 False
# None 4 False
# None 5 False
# None 6 False
# 1 7 True
```

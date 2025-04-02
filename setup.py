#!/usr/bin/env python

# Copyright 2024 A2labs, Inc.
# All rights reserved.
#

from setuptools import setup, find_packages, Extension
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py

import importlib.util
from pathlib import Path

package_name = 'gpuinfo'
description = ''
author = 'Mako'
author_email = 'lukasz@mako-dev.com'
url = 'https://github.com/makodevai/gpuinfo'
download_url = 'https://github.com/makodevai/gpuinfo'
data_files = {}

version_file = Path(__file__).parent.joinpath(package_name, 'version.py')
spec = importlib.util.spec_from_file_location('{}.version'.format(package_name), version_file)
package_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_version)

long_desc = None
long_desc_type = None
readme_md = Path(__file__).parent.joinpath('README.md')
if readme_md.exists():
    data_files.setdefault('', []).append(readme_md.name)
    with readme_md.open('r') as f:
        long_desc = f.read()
        long_desc_type = 'text/markdown'

license = Path(__file__).parent.joinpath('LICENSE')
if license.exists():
    data_files.setdefault('', []).append(license.name)

data_files.setdefault('', []).append(str(Path(__file__).parent.joinpath('gpuinfo', 'csrc', 'types.h')))


class dist_info_mixin:
    def run(self):
        _dist_file = version_file.parent.joinpath('_dist_info.py')
        _dist_file.write_text('\n'.join(
            map(lambda attr_name: attr_name + ' = ' + repr(getattr(package_version, attr_name)),
                package_version.__all__)) + '\n')
        try:
            ret = super().run()
        finally:
            _dist_file.unlink()
        return ret


class custom_sdist(dist_info_mixin, sdist):
    pass


class custom_wheel(dist_info_mixin, build_py):
    pass


setup(name=package_name,
      version=package_version.version,
      description=description,
      author=author,
      author_email=author_email,
      url=url,
      download_url=download_url,
      long_description=long_desc,
      long_description_content_type=long_desc_type,
      python_requires='>=3.10.0',
      setup_requires=[
          'GitPython'
      ],
      install_requires=[],
      packages=find_packages(where='.', include=['gpuinfo', 'gpuinfo.*']),
      package_data=data_files,
      package_dir={ '': '.' },
      cmdclass={
          'sdist': custom_sdist,
          'build_py': custom_wheel
      },
      ext_modules=[
          Extension("gpuinfo.C", ["gpuinfo/csrc/gpuinfo.c", "gpuinfo/csrc/amd.c", "gpuinfo/csrc/cuda.c"], extra_compile_args=['-O0', '-g'])
      ],
)

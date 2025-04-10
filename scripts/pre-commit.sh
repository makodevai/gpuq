#!/usr/bin/env bash

set -xe

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment"
    exit 1
fi

if [ "$1" == "-f" ]; then
  black gpuinfo
else
  black --check gpuinfo
fi;

mypy --strict gpuinfo

pytest tests/

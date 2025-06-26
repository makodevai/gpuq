#!/usr/bin/env bash

set -xe

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment"
    exit 1
fi

if [ "$1" == "-f" ]; then
  black gpuinfo tests
else
  black --check gpuinfo test
fi;

mypy --strict gpuinfo tests

pytest tests/

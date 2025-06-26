#!/usr/bin/env bash

set -xe

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate the virtual environment"
    exit 1
fi

if [ "$1" == "-f" ]; then
  black gpuq tests
else
  black --check gpuq test
fi;

mypy --strict gpuq tests

pytest tests/

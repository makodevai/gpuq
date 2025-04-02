#!/usr/bin/env bash

set -xe

if [ "$GITHUB_CI" == 1 ]; then
  echo "Running on GitHub CI"
else
  echo "Running on local machine"
fi

if [ "$GITHUB_CI" == 1 ]; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-pip python3-venv
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

pip install -e '.[dev]'

mkdir -p junit

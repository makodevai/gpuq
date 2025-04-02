#!/usr/bin/env bash

set -xe

source .venv/bin/activate

mypy mako_engine --junit-xml=junit/mypy-results.xml

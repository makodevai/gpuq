#!/usr/bin/env bash

set -xe

source .venv/bin/activate

pytest tests/ --junitxml=junit/test-results.xml

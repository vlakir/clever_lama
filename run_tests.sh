#!/bin/bash

poetry run pytest --import-mode=importlib "$@"
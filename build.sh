#!/bin/bash

docker compose down --volumes --remove-orphans

docker system prune -a

docker compose build --no-cache

docker compose up
#!/bin/bash

docker compose down --volumes --remove-orphans

docker system prune -a

docker compose build --no-cache

docker tag clever-lama vlakir/clever-lama:latest

docker push vlakir/clever-lama:latest
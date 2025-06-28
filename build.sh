#!/bin/bash

# Остановить и удалить все контейнеры проекта
docker-compose down --volumes --remove-orphans

# Удалить все неиспользуемые образы
docker system prune -a

# Пересобрать образ с нуля
docker-compose build --no-cache

# Запустить заново
docker-compose up
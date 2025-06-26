# CleverLama

Мост между OpenAI-совместимыми провайдерами и Ollama API

## Описание

CleverLama - это API-мост, который позволяет использовать OpenAI-совместимые провайдеры 
(например, через API aitunnel.ru) с приложениями, ожидающими Ollama API (например, 
JetBrains AI Assistant).


## Возможности

- ✅ Совместимость с Ollama API
- ✅ Поддержка OpenAI-совместимых провайдеров
- ✅ Трансформация запросов между форматами
- ✅ Поддержка streaming ответов
- ✅ Docker контейнеризация
- ✅ Логирование и мониторинг
- ✅ CLI для управления

## Быстрый старт

### С помощью Docker Compose

1. Клонируйте репозиторий:
```bash
git clone https://github.com/vlakir/clever-lama.git
cd clever-lama
```

2. Создайте файл `.env` с вашими настройками:
```bash
API_KEY=your_api_key_here
API_BASE_URL=https://api.aitunnel.ru/v1
LOG_LEVEL=INFO
```

3. Запустите сервис:
```bash
docker-compose up -d
```

### Локальная разработка

1. Установите Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Установите зависимости:
```bash
poetry install
```

3. Запустите сервер:
```bash
poetry run python src/clever_lama/main.py
```

## Использование

После запуска сервис будет доступен на `http://localhost:11434` и предоставляет следующие эндпоинты:

- `GET /` - проверка здоровья сервиса
- `GET /api/version` - версия API
- `GET /api/tags` - список доступных моделей
- `POST /api/show` - информация о модели
- `POST /api/pull` - загрузка модели
- `POST /api/generate` - генерация текста
- `POST /api/chat` - чат с моделью

## Конфигурация

Настройки задаются через переменные окружения:

- `API_KEY` - ключ API для провайдера (обязательно)
- `API_BASE_URL` - базовый URL API провайдера (по умолчанию: https://api.aitunnel.ru/v1)
- `LOG_LEVEL` - уровень логирования (по умолчанию: INFO)
- `CACHE_DURATION_MINUTES` - время кеширования в минутах (по умолчанию: 10)
- `HOST` - хост для привязки (по умолчанию: 0.0.0.0)
- `PORT` - порт для привязки (по умолчанию: 11434)

## CLI команды

После установки доступны следующие команды:

- `clever-lama-manage` - основная команда управления
- `clever-lama-dev` - запуск в режиме разработки
- `clever-lama-prod` - запуск в продакшн режиме
- `clever-lama-build` - сборка Docker образа

## Разработка

### Требования

- Python 3.11+
- Poetry
- Docker (опционально)

### Установка для разработки

```bash
poetry install --with dev
```

### Запуск тестов

```bash
poetry run pytest
```

### Линтинг и форматирование

```bash
./check.sh
```

## Лицензия

MIT License

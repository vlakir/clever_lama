# Dockerfile
FROM python:3.13.3-slim

# Создаем пользователя для безопасности
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry
RUN pip install poetry

# Настройки Poetry - НЕ создаем виртуальное окружение!
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONPATH=/app/src

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы Poetry
COPY pyproject.toml poetry.lock* ./

# Копируем README.md чтобы Poetry не ругался
COPY README.md ./

# Устанавливаем зависимости через Poetry (БЕЗ pip!)
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Копируем исходный код
COPY src/ ./src/

# Меняем владельца файлов
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:11434/ || exit 1

# Открываем порт
EXPOSE 11434

# Запуск через Python с правильным путем
CMD ["python", "src/clever_lama/main.py"]

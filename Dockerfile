# Dockerfile
ARG PYTHON_VERSION=3.13-slim
FROM python:${PYTHON_VERSION} as builder

# 1. Установка зависимостей с очисткой кеша в одном слое
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Установка Poetry официальным методом
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python - && \
    mv /root/.local/bin/poetry /usr/local/bin/

# 3. Оптимизированные настройки Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PIP_NO_CACHE_DIR=off

WORKDIR /app
COPY pyproject.toml poetry.lock* README.md ./

# 4. Установка зависимостей без корневого доступа
RUN poetry install --only=main --no-root && \
    rm -rf "$POETRY_CACHE_DIR"

# 5. Финальный образ
FROM python:${PYTHON_VERSION}
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages/ /usr/local/lib/python3.13/site-packages/
COPY src/ ./src/

USER appuser
EXPOSE 11434
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:11434/ || exit 1
CMD ["python", "src/clever_lama/main.py"]
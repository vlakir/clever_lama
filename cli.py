"""Command line interface for CleverLama."""

import subprocess
import sys
from pathlib import Path

import click
import requests
from rich.console import Console
from rich.table import Table

from src.clever_lama import __version__
from src.clever_lama.constants import RESPONSE_CODE_OK

console = Console()


def run_command_streaming(command: str, *, shell: bool = True) -> int:
    """Выполняет команду с потоковым выводом."""
    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Читаем вывод построчно и выводим в консоль
        for line in process.stdout:
            print(line.rstrip())

        process.wait()
        return process.returncode

    except Exception as e:
        console.print(f'❌ Ошибка выполнения команды: {e}', style='red')
        return 1


def check_docker() -> bool:
    """Проверяет доступность Docker."""
    try:
        check_result = subprocess.run(
            ['docker', '--version'],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )

        if check_result.returncode == 0:
            return True
        console.print('❌ Docker не доступен', style='red')

    except FileNotFoundError:
        console.print('❌ Docker не установлен', style='red')
        return False
    else:
        return False


def check_docker_compose() -> bool:
    """Проверяет доступность Docker Compose."""
    try:
        check_result = subprocess.run(
            ['docker', 'compose', 'version'],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )

        if check_result.returncode == 0:
            return True
        console.print('❌ Docker Compose не доступен', style='red')

    except FileNotFoundError:
        console.print('❌ Docker Compose не установлен', style='red')
        return False
    else:
        return False


@click.group()
def cli() -> None:
    """CleverLama CLI - инструмент для управления API мостом."""


@cli.command()
def start() -> None:
    """Запускает сервис через Docker Compose."""
    if not check_docker() or not check_docker_compose():
        sys.exit(1)

    console.print('🚀 Запускаем CleverLama...', style='green')
    returncode = run_command_streaming('docker compose up -d')

    if returncode == 0:
        console.print('✅ CleverLama успешно запущен', style='green')
        console.print(
            '📡 Сервис доступен по адресу: http://localhost:11434', style='blue'
        )
    else:
        console.print('❌ Ошибка запуска CleverLama', style='red')
        sys.exit(1)


@cli.command()
def stop() -> None:
    """Останавливает сервис."""
    if not check_docker() or not check_docker_compose():
        sys.exit(1)

    console.print('🛑 Останавливаем CleverLama...', style='yellow')
    returncode = run_command_streaming('docker compose down')

    if returncode == 0:
        console.print('✅ CleverLama успешно остановлен', style='green')
    else:
        console.print('❌ Ошибка остановки CleverLama', style='red')
        sys.exit(1)


@cli.command()
def restart() -> None:
    """Перезапускает сервис."""
    if not check_docker() or not check_docker_compose():
        sys.exit(1)

    console.print('🔄 Перезапускаем CleverLama...', style='yellow')

    # Останавливаем
    returncode = run_command_streaming('docker compose down')
    if returncode != 0:
        console.print('❌ Ошибка остановки CleverLama', style='red')
        sys.exit(1)

    # Запускаем
    returncode = run_command_streaming('docker compose up -d')
    if returncode == 0:
        console.print('✅ CleverLama успешно перезапущен', style='green')
        console.print(
            '📡 Сервис доступен по адресу: http://localhost:11434', style='blue'
        )
    else:
        console.print('❌ Ошибка запуска CleverLama', style='red')
        sys.exit(1)


@cli.command()
def status() -> None:
    """Показывает статус сервиса."""
    if not check_docker() or not check_docker_compose():
        sys.exit(1)

    console.print('📊 Статус CleverLama:', style='blue')
    run_command_streaming('docker compose ps')


@cli.command()
def logs() -> None:
    """Показывает логи сервиса."""
    if not check_docker() or not check_docker_compose():
        sys.exit(1)

    console.print('📝 Логи CleverLama:', style='blue')
    run_command_streaming('docker compose logs -f --tail=50')


@cli.command()
@click.option(
    '--follow', '-f', is_flag=True, help='Следить за логами в реальном времени'
)
@click.option('--tail', '-n', default=50, help='Количество последних строк логов')
def tail(*, follow: bool, tail_: int) -> None:
    """Показывает последние логи сервиса."""
    if not check_docker() or not check_docker_compose():
        sys.exit(1)

    console.print(f'📝 Последние {tail_} строк логов CleverLama:', style='blue')

    command = f'docker compose logs --tail={tail_}'
    if follow:
        command += ' -f'

    run_command_streaming(command)


@cli.command()
def build() -> None:
    """Собирает Docker образ заново."""
    if not check_docker() or not check_docker_compose():
        sys.exit(1)

    console.print('🔨 Пересобираем CleverLama...', style='blue')
    returncode = run_command_streaming('docker compose build --no-cache')

    if returncode == 0:
        console.print('✅ CleverLama успешно пересобран', style='green')
    else:
        console.print('❌ Ошибка сборки CleverLama', style='red')
        sys.exit(1)


@cli.command()
def clean() -> None:
    """Очищает Docker ресурсы."""
    if not check_docker() or not check_docker_compose():
        sys.exit(1)

    console.print('🧹 Очищаем Docker ресурсы...', style='yellow')

    # Останавливаем и удаляем контейнеры
    returncode = run_command_streaming(
        'docker compose down --rmi all --volumes --remove-orphans')

    if returncode == 0:
        console.print('✅ Docker ресурсы очищены', style='green')
    else:
        console.print('❌ Ошибка очистки Docker ресурсов', style='red')
        sys.exit(1)


@cli.command()
def test() -> None:
    """Тестирует подключение к API."""
    console.print('🧪 Тестируем API CleverLama...', style='blue')

    try:
        # Тестируем health check
        response = requests.get('http://localhost:11434/', timeout=10)
        if response.status_code == RESPONSE_CODE_OK:
            console.print('✅ Health check прошел успешно', style='green')
        else:
            console.print(
                f'❌ Health check failed: {response.status_code}', style='red'
            )
            return

        # Тестируем получение моделей
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == RESPONSE_CODE_OK:
            models = response.json()
            console.print('✅ API моделей работает', style='green')
            console.print(
                f'📦 Доступно моделей: {len(models.get("models", []))}', style='blue'
            )
        else:
            console.print(
                f'❌ API моделей не работает: {response.status_code}', style='red'
            )

    except requests.RequestException as e:
        console.print(f'❌ Ошибка подключения к API: {e}', style='red')


@cli.command()
def config() -> None:
    """Показывает текущую конфигурацию."""
    console.print('⚙️ Конфигурация CleverLama:', style='blue')

    # Читаем .env файл если есть
    env_file = Path('.env')
    if env_file.exists():
        with env_file.open('r', encoding='utf-8') as f:
            env_content = f.read()

        table = Table(title='Переменные окружения')
        table.add_column('Переменная', style='cyan')
        table.add_column('Значение', style='green')

        for line in env_content.strip().split('\n'):
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Скрываем чувствительные данные
                if (
                        'KEY' in key.upper()
                        or 'PASSWORD' in key.upper()
                        or 'SECRET' in key.upper()
                ):
                    value = '***СКРЫТО***'
                table.add_row(key, value)

        console.print(table)
    else:
        console.print('📄 Файл .env не найден', style='yellow')


@cli.command()
def info() -> None:
    """Показывает информацию о системе."""
    table = Table(title='Информация о системе')
    table.add_column('Параметр', style='cyan')
    table.add_column('Значение', style='green')

    # Версия CleverLama
    table.add_row('CleverLama версия', __version__)

    # Версия Python
    table.add_row('Python версия', sys.version.split()[0])
    table.add_row('Python путь', sys.executable)

    # Docker версии
    try:
        docker_result = subprocess.run(
            ['docker', '--version'],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if docker_result.returncode == 0:
            docker_version = docker_result.stdout.strip()
            table.add_row('Docker', docker_version)
        else:
            table.add_row('Docker', '❌ Не доступен')
    except FileNotFoundError:
        table.add_row('Docker', '❌ Не установлен')

    try:
        compose_result = subprocess.run(
            ['docker', 'compose', 'version'],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if compose_result.returncode == 0:
            compose_version = compose_result.stdout.strip()
            table.add_row('Docker Compose', compose_version)
        else:
            table.add_row('Docker Compose', '❌ Не доступен')
    except FileNotFoundError:
        table.add_row('Docker Compose', '❌ Не установлен')

    # Информация о Python
    table.add_row('Python версия', sys.version.split()[0])
    table.add_row('Python путь', sys.executable)

    console.print(table)


if __name__ == '__main__':
    cli()
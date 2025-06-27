import logging
import sys
from types import MappingProxyType

from config import settings


class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для логов."""

    # ANSI escape коды для цветов
    COLORS = MappingProxyType(
        {
            'DEBUG': '\033[36m',  # Cyan
            'INFO': '\033[0m',  # Default/White (вместо красного)
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',  # Red (только ERROR будет красным)
            'CRITICAL': '\033[35m',  # Magenta
            'RESET': '\033[0m',  # Reset
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        # Получаем исходное сообщение
        log_message = super().format(record)

        # Добавляем цвет в зависимости от уровня
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        return f'{color}{log_message}{reset}'


# Настройка логирования с цветным форматтером
def setup_logging() -> None:
    """Настраивает логирование с цветным выводом."""
    # Создаем форматтер
    formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Получаем root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Удаляем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Создаем новый обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Добавляем обработчик к root logger
    root_logger.addHandler(console_handler)


# Инициализируем логирование
setup_logging()
logger = logging.getLogger(__name__)

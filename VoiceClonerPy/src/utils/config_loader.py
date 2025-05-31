import yaml
import os
import logging # Используем logging для единообразия

# Настройка логгера для этого модуля
logger = logging.getLogger(__name__)
# Уровень и обработчик будут настроены в основном скрипте run.py или при инициализации Trainer
# Если этот модуль используется отдельно, можно добавить базовую конфигурацию:
# if not logger.handlers:
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load_config(config_path):
    """
    Загружает конфигурационный файл YAML.

    Args:
        config_path (str): Путь к конфигурационному файлу YAML.

    Returns:
        dict: Загруженная конфигурация в виде словаря.
    Raises:
        FileNotFoundError: Если файл конфигурации не найден.
        yaml.YAMLError: Если произошла ошибка при парсинге YAML.
        Exception: Другие неожиданные ошибки при загрузке.
    """
    if not os.path.exists(config_path):
        logger.error(f"Файл конфигурации не найден по пути: {config_path}")
        raise FileNotFoundError(f"Файл конфигурации не найден по пути: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Конфигурация успешно загружена из {config_path}")
        return config_dict
    except yaml.YAMLError as e:
        logger.error(f"Ошибка парсинга YAML файла {config_path}: {e}")
        raise  # Перевыбрасываем исключение для обработки выше
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке конфигурации {config_path}: {e}")
        raise # Перевыбрасываем исключение

def save_config(config_dict, config_path):
    """
    Сохраняет словарь в конфигурационный файл YAML.

    Args:
        config_dict (dict): Словарь конфигурации для сохранения.
        config_path (str): Путь для сохранения файла YAML.

    Returns:
        bool: True в случае успеха, False в противном случае.
    """
    try:
        # Убедимся, что родительская директория существует
        parent_dir = os.path.dirname(config_path)
        if parent_dir: # Если путь не просто имя файла
             os.makedirs(parent_dir, exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        logger.info(f"Конфигурация сохранена в {config_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения конфигурации в {config_path}: {e}")
        return False

import argparse
import torch
import os
import logging
import time
import sys
import yaml

# Импорт модулей проекта
from src.utils.config_loader import load_config
from src.training.trainer import Trainer
from src.models.stargan_vc import Generator, Discriminator
from src.data_loader import VoiceDataset, DataLoader, prepare_base_model_data, load_data_for_dataset # load_data_for_dataset was added in prev step
from src.inference import convert_voice_from_file
from src.realtime_audio_utils import list_audio_devices, select_device_id
from src.realtime_inference import RealTimeVoiceConverter

# Настройка базового логгера для скрипта run.py
logger = logging.getLogger("VoiceClonerPy_RunScript") # Имя логгера
logger.setLevel(logging.INFO) # Уровень логирования
if not logger.handlers: # Предотвращение добавления нескольких обработчиков при повторном запуске
    ch_run = logging.StreamHandler(sys.stdout) # Явно используем stdout
    formatter_run = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch_run.setFormatter(formatter_run)
    logger.addHandler(ch_run)

def main(args):
    print(f"VoiceClonerPy запускается...") # Начальное сообщение о запуске
    print(f"Используется конфигурационный файл: {args.config}")

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Конфигурационный файл не найден: {args.config}. Выход.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Ошибка парсинга конфигурационного файла {args.config}: {e}. Выход.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке конфигурации {args.config}: {e}. Выход.")
        sys.exit(1)

    if not config:
        logger.error(f"Не удалось загрузить конфигурацию из {args.config} (вернулся None). Выход.")
        sys.exit(1)

    # Определение эффективного режима работы
    effective_mode = config['training'].get('mode', 'train') # По умолчанию 'train', если не указано в конфиге
    if args.mode:
        effective_mode = args.mode # CLI переопределяет конфиг
        config['training']['mode'] = effective_mode # Обновляем состояние конфига, если переопределено

    print(f"Режим работы: {effective_mode}")
    logger.info(f"Текущий проект: {config['project'].get('name', 'N/A')}, Эксперимент: {config['project'].get('experiment_name', 'N/A')}")
    logger.info(f"Запуск в режиме: {effective_mode}")

    # Настройка устройства (общее для большинства режимов)
    device_name = config['training'].get('device', 'cpu')
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        if device_name == 'cuda':
            logger.warning("CUDA указана в конфигурации, но недоступна. Используется CPU.")
        device = torch.device('cpu')
    logger.info(f"Используемое устройство: {device}")

    # Инициализация моделей
    try:
        generator = Generator(config=config['model']).to(device)
        discriminator = Discriminator(num_speakers=config['model']['num_speakers'], **config['model']['discriminator']).to(device)
    except KeyError as e:
        logger.error(f"Ошибка конфигурации модели: отсутствует ключ {e}. Проверьте ваш конфигурационный файл. Выход.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка инициализации моделей: {e}. Выход.")
        sys.exit(1)

    # --- Выполнение в зависимости от режима ---
    if effective_mode == 'train' or effective_mode == 'fine_tune':
        logger.info(f"Настройка для режима: {effective_mode}...")

        data_dir_key = 'fine_tune_user_data_dir' if effective_mode == 'fine_tune' else 'base_model_corpus_dir'
        metadata_key = 'fine_tune_metadata' if effective_mode == 'fine_tune' else 'base_model_metadata' # Assuming these keys might exist

        data_dir = config['data'].get(data_dir_key)
        metadata = config['data'].get(metadata_key) # This could be None

        if not data_dir:
            logger.error(f"Директория данных '{data_dir_key}' не указана в конфигурации для режима '{effective_mode}'. Выход.")
            sys.exit(1)

        train_data_entries = load_data_for_dataset(data_dir, metadata, config['model']['num_speakers'], is_validation=False)
        train_dataset = VoiceDataset(train_data_entries, config['data'], is_validation=False)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True,
                                  num_workers=config['data']['num_workers'], pin_memory=config['data']['pin_memory'])

        val_loader = None
        if config['data'].get('val_data_dir') and config['data'].get('val_metadata_file'):
            val_data_entries = load_data_for_dataset(
                config['data']['val_data_dir'], config['data']['val_metadata_file'],
                config['model']['num_speakers'], is_validation=True
            )
            if val_data_entries:
                val_dataset = VoiceDataset(val_data_entries, config['data'], is_validation=True)
                val_loader = DataLoader(val_dataset, batch_size=config['training'].get('batch_size_val', 4), shuffle=False,
                                        num_workers=config['data']['num_workers'], pin_memory=config['data']['pin_memory'])
                logger.info(f"Валидационный датасет загружен с {len(val_dataset)} сэмплами.")
            else: logger.warning("Не удалось загрузить записи для валидационного датасета (load_data_for_dataset вернул пустой список).")
        else: logger.info("Директория валидационных данных или файл метаданных не указаны. Валидация пропускается.")

        if effective_mode == 'fine_tune':
            ft_conf = config.get('fine_tuning', {})
            config['training']['num_epochs'] = ft_conf.get('num_epochs', config['training']['num_epochs'])
            config['training']['learning_rate_g'] = ft_conf.get('learning_rate_g', config['training']['learning_rate_g'])
            config['training']['learning_rate_d'] = ft_conf.get('learning_rate_d', config['training']['learning_rate_d'])

            base_model_g_path = ft_conf.get('base_model_checkpoint_path')
            if base_model_g_path and os.path.exists(base_model_g_path):
                logger.info(f"Загрузка базового генератора из: {base_model_g_path} для fine-tuning.")
                try:
                    gen_ckpt = torch.load(base_model_g_path, map_location=device)
                    if 'generator_state_dict' in gen_ckpt: generator.load_state_dict(gen_ckpt['generator_state_dict'])
                    else: generator.load_state_dict(gen_ckpt)
                    logger.info("Базовый генератор загружен для fine-tuning.")
                except Exception as e: logger.error(f"Ошибка загрузки чекпоинта базового генератора {base_model_g_path}: {e}.")
            else: logger.warning(f"Чекпоинт базового генератора для fine-tuning не найден или не указан: {base_model_g_path}.")

        trainer = Trainer(config, generator, discriminator, train_loader, val_loader, device)
        logger.info(f"Trainer для режима '{effective_mode}' инициализирован. Запуск процесса...")
        trainer.run_training()
        logger.info(f"Режим '{effective_mode}' завершен.")

    elif effective_mode == 'inference_file':
        logger.info("Настройка для режима 'inference_file'...")
        inf_conf = config.get('inference_file')
        if not inf_conf: logger.error("Секция `inference_file` отсутствует в конфигурации. Выход."); sys.exit(1)

        model_path_to_load, target_speaker_id = None, 0
        direction = inf_conf.get('conversion_direction', 'specific')

        if direction == 'male_to_female':
            model_path_to_load = inf_conf.get('fine_tuned_model_path_female_generator')
            target_speaker_id = inf_conf.get('target_female_speaker_id', 0)
        elif direction == 'female_to_male':
            model_path_to_load = inf_conf.get('fine_tuned_model_path_male_generator')
            target_speaker_id = inf_conf.get('target_male_speaker_id', 1)
        elif direction == 'specific':
            model_path_to_load = inf_conf.get('specific_generator_checkpoint_path')
            target_speaker_id = inf_conf.get('specific_target_speaker_id', 0)
        else:
            logger.error(f"Неверное значение 'conversion_direction' в конфигурации: {direction}. Выход.")
            sys.exit(1)

        if not model_path_to_load:
             logger.error(f"Путь к модели для инференса не указан для направления '{direction}'. Выход.")
             sys.exit(1)
        if not os.path.exists(model_path_to_load):
             logger.error(f"Файл модели для инференса не найден: {model_path_to_load}. Выход.")
             sys.exit(1)

        convert_voice_from_file(config, model_path_to_load, inf_conf['input_wav_path'],
                                inf_conf['output_wav_path'], target_speaker_id)

    elif effective_mode == 'inference_realtime':
        logger.info("Настройка для режима 'inference_realtime'...")
        rt_conf = config.get('inference_realtime')
        if not rt_conf: logger.error("Секция `inference_realtime` отсутствует в конфигурации. Выход."); sys.exit(1)

        list_audio_devices()

        input_dev_idx = rt_conf.get('input_device_index', -1)
        output_dev_idx = rt_conf.get('output_device_index', -1)

        if input_dev_idx == -1 :
            logger.info("ID устройства ввода не указан (-1) в конфигурации. Требуется интерактивный выбор.")
            input_dev_idx = select_device_id("Выберите ID устройства ВВОДА: ", kind="ввода")
            if input_dev_idx is None: logger.error("Устройство ввода не выбрано. Выход."); sys.exit(1)
            rt_conf['input_device_index'] = input_dev_idx

        if output_dev_idx == -1:
            logger.info("ID устройства вывода не указан (-1) в конфигурации. Требуется интерактивный выбор.")
            output_dev_idx = select_device_id("Выберите ID устройства ВЫВОДА: ", kind="вывода")
            if output_dev_idx is None: logger.error("Устройство вывода не выбрано. Выход."); sys.exit(1)
            rt_conf['output_device_index'] = output_dev_idx

        logger.info(f"Используется ID устройства ввода: {rt_conf['input_device_index']}")
        logger.info(f"Используется ID устройства вывода: {rt_conf['output_device_index']}")

        try:
            converter = RealTimeVoiceConverter(config)
            converter.start()
        except KeyboardInterrupt:
            logger.info("Преобразование в реальном времени остановлено пользователем (Ctrl+C).")
        except Exception as e:
            logger.error(f"Ошибка во время инференса в реальном времени: {e}", exc_info=True)
        finally:
            if 'converter' in locals() and hasattr(converter, 'stop'):
                converter.stop()
        logger.info("Сессия инференса в реальном времени завершена.")

    else:
        logger.error(f"Неизвестный или неподдерживаемый режим: '{effective_mode}'. Проверьте конфигурацию или аргумент --mode. Выход.")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="VoiceClonerPy: Система для клонирования голоса на основе StarGAN-VC. "
                    "Поддерживает обучение, тонкую настройку и инференс (из файла и в реальном времени)."
    )
    parser.add_argument(
        '--config',
        type=str,
        default="config/config_template.yaml",
        help="Путь к YAML файлу конфигурации (по умолчанию: config/config_template.yaml)."
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'fine_tune', 'inference_file', 'inference_realtime'],
        help="Переопределить режим работы (train, fine_tune, inference_file, inference_realtime)."
    )

    cmd_args = parser.parse_args()
    main(cmd_args)

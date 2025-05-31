import argparse
import torch
import os
import logging
import time
import sys
import yaml
import csv # Для чтения CSV в load_data_for_dataset

# Импорт модулей проекта
from VoiceClonerPy.src.utils.config_loader import load_config
from VoiceClonerPy.src.training.trainer import Trainer
from VoiceClonerPy.src.models.stargan_vc import Generator, Discriminator
from VoiceClonerPy.src.data_loader import VoiceDataset, DataLoader # prepare_base_model_data убрано, т.к. load_data_for_dataset его заменяет
from VoiceClonerPy.src.inference import convert_voice_from_file
from VoiceClonerPy.src.realtime_audio_utils import list_audio_devices, select_device_id
from VoiceClonerPy.src.realtime_inference import RealTimeVoiceConverter

logger = logging.getLogger("VoiceClonerPy_RunScript")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch_run = logging.StreamHandler(sys.stdout)
    formatter_run = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch_run.setFormatter(formatter_run)
    logger.addHandler(ch_run)

def load_data_for_dataset(data_root_path, metadata_file_path, num_expected_speakers, is_validation=False, dataset_type='train'):
    """
    Загружает метаданные и формирует список записей для VoiceDataset.
    Args:
        data_root_path (str): Корневой путь к данным (может быть не использован, если wav_path в CSV абсолютные).
        metadata_file_path (str): Путь к CSV файлу с метаданными.
                                  Ожидаемый формат CSV: wav_path,speaker_id,speaker_name
                                  Для валидации, если is_validation=True, может быть другой формат,
                                  но для этого теста мы упрощаем и используем тот же формат.
        num_expected_speakers (int): Ожидаемое количество дикторов (для информации).
        is_validation (bool): Флаг, указывающий, загружаются ли данные для валидации.
        dataset_type (str): 'train' или 'val', для логирования.
    Returns:
        list: Список кортежей (wav_path, speaker_id_int)
    """
    logger.info(f"Загрузка данных для набора '{dataset_type}' из метафайла: {metadata_file_path}")
    data_entries = []
    if not metadata_file_path or not os.path.exists(metadata_file_path):
        logger.warning(f"Файл метаданных не найден или не указан: {metadata_file_path}. Возвращаем пустой список для '{dataset_type}'.")
        return data_entries

    speaker_ids_found = set()
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not all(col in reader.fieldnames for col in ['wav_path', 'speaker_id']):
                logger.error(f"Необходимые колонки 'wav_path', 'speaker_id' отсутствуют в {metadata_file_path}. Выход.")
                # В реальном приложении здесь может быть sys.exit(1) или raise Error
                return [] # Возвращаем пустой список при ошибке

            for row in reader:
                wav_path = row['wav_path']
                # Если путь в CSV относительный, он должен быть относительно data_root_path
                # Для данного теста предполагаем, что run_pipeline_test.py создаст абсолютные пути или пути относительно корня проекта.
                if data_root_path and not os.path.isabs(wav_path): # Это условие может потребовать доработки в зависимости от структуры CSV
                    wav_path = os.path.join(data_root_path, wav_path)

                try:
                    speaker_id = int(row['speaker_id'])
                    speaker_ids_found.add(speaker_id)
                except ValueError:
                    logger.warning(f"Неверный speaker_id '{row['speaker_id']}' для файла {wav_path} в {metadata_file_path}. Пропуск.")
                    continue

                # Для этого упрощенного теста VoiceDataset ожидает (wav_path, speaker_id_int)
                data_entries.append((wav_path, speaker_id))

        logger.info(f"Загружено {len(data_entries)} записей для набора '{dataset_type}'. Найдено уникальных ID дикторов: {len(speaker_ids_found)} (Макс ID: {max(speaker_ids_found) if speaker_ids_found else 'N/A'}).")
        if num_expected_speakers and max(speaker_ids_found if speaker_ids_found else [-1]) >= num_expected_speakers:
             logger.warning(f"Максимальный ID диктора ({max(speaker_ids_found)}) >= num_speakers ({num_expected_speakers}) из конфига. Это может вызвать ошибку Embedding слоя.")

    except Exception as e:
        logger.error(f"Ошибка при чтении или обработке метафайла {metadata_file_path}: {e}", exc_info=True)
        return [] # Возвращаем пустой список при ошибке

    return data_entries


def main(args):
    print(f"VoiceClonerPy запускается...")
    print(f"Используется конфигурационный файл: {args.config}")

    try: config = load_config(args.config)
    except FileNotFoundError: logger.error(f"Конфигурационный файл не найден: {args.config}. Выход."); sys.exit(1)
    except yaml.YAMLError as e: logger.error(f"Ошибка парсинга конфигурационного файла {args.config}: {e}. Выход."); sys.exit(1)
    except Exception as e: logger.error(f"Неожиданная ошибка при загрузке конфигурации {args.config}: {e}. Выход."); sys.exit(1)
    if not config: logger.error(f"Не удалось загрузить конфигурацию из {args.config}. Выход."); sys.exit(1)

    effective_mode = config['training'].get('mode', 'train')
    if args.mode: effective_mode = args.mode; config['training']['mode'] = effective_mode

    print(f"Режим работы: {effective_mode}")
    logger.info(f"Проект: {config['project'].get('name', 'N/A')}, Эксперимент: {config['project'].get('experiment_name', 'N/A')}, Режим: {effective_mode}")

    device_name = config['training'].get('device', 'cpu')
    device = torch.device('cuda' if device_name == 'cuda' and torch.cuda.is_available() else 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available(): logger.warning("CUDA указана, но недоступна. Используется CPU.")
    logger.info(f"Используемое устройство: {device}")

    try:
        generator = Generator(config=config['model']).to(device)
        discriminator = Discriminator(num_speakers=config['model']['num_speakers'], **config['model']['discriminator']).to(device)
    except KeyError as e: logger.error(f"Ошибка конфигурации модели: ключ {e}. Выход."); sys.exit(1)
    except Exception as e: logger.error(f"Ошибка инициализации моделей: {e}. Выход."); sys.exit(1)

    if effective_mode == 'train' or effective_mode == 'fine_tune':
        logger.info(f"Настройка для режима: {effective_mode}...")

        data_root = config['data'].get('base_data_path', '.') # Общий корень для данных, если пути в CSV относительные

        train_metadata_path = config['data'].get('train_metadata_file')
        val_metadata_path = config['data'].get('val_metadata_file')

        if not train_metadata_path:
            logger.error(f"Отсутствует 'train_metadata_file' в конфигурации данных для режима '{effective_mode}'. Выход.")
            sys.exit(1)

        train_data_entries = load_data_for_dataset(data_root, train_metadata_path, config['model']['num_speakers'], is_validation=False, dataset_type='train')
        if not train_data_entries: logger.error(f"Не удалось загрузить данные для обучения. Проверьте метафайл: {train_metadata_path}. Выход."); sys.exit(1)

        train_dataset = VoiceDataset(config['data'], train_data_entries, is_validation=False)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True,
                                  num_workers=config['data']['num_workers'], pin_memory=config['data']['pin_memory'])

        val_loader = None
        if val_metadata_path: # val_data_dir больше не используется напрямую здесь, только val_metadata_file
            val_data_entries = load_data_for_dataset(data_root, val_metadata_path, config['model']['num_speakers'], is_validation=True, dataset_type='val')
            if val_data_entries:
                val_dataset = VoiceDataset(config['data'], val_data_entries, is_validation=True)
                val_loader = DataLoader(val_dataset, batch_size=config['training'].get('batch_size_val', 4), shuffle=False,
                                        num_workers=config['data']['num_workers'], pin_memory=config['data']['pin_memory'])
                logger.info(f"Валидационный датасет загружен с {len(val_dataset)} сэмплами.")
            else: logger.warning("Не удалось загрузить записи для валидационного датасета. Валидация будет пропущена.")
        else: logger.info("Файл метаданных для валидации ('val_metadata_file') не указан. Валидация пропускается.")

        if effective_mode == 'fine_tune':
            # ... (логика fine-tune как в предыдущем шаге) ...
            ft_conf = config.get('fine_tuning', {})
            config['training']['num_epochs'] = ft_conf.get('num_epochs', config['training']['num_epochs'])
            base_model_g_path = ft_conf.get('base_model_checkpoint_path')
            if base_model_g_path and os.path.exists(base_model_g_path):
                logger.info(f"Загрузка базового генератора из: {base_model_g_path} для fine-tuning.")
                try:
                    gen_ckpt = torch.load(base_model_g_path, map_location=device);
                    if 'generator_state_dict' in gen_ckpt: generator.load_state_dict(gen_ckpt['generator_state_dict'])
                    else: generator.load_state_dict(gen_ckpt)
                except Exception as e: logger.error(f"Ошибка загрузки чекпоинта: {e}")
            else: logger.warning(f"Чекпоинт для fine-tuning не найден: {base_model_g_path}")

        trainer = Trainer(config, generator, discriminator, train_loader, val_loader, device)
        logger.info(f"Trainer для режима '{effective_mode}' инициализирован. Запуск процесса...")
        trainer.run_training()
        logger.info(f"Режим '{effective_mode}' завершен.")

    elif effective_mode == 'inference_file':
        # ... (логика inference_file как в предыдущем шаге) ...
        logger.info("Настройка для режима 'inference_file'...")
        inf_conf = config.get('inference_file');
        if not inf_conf: logger.error("Секция `inference_file` отсутствует. Выход."); sys.exit(1)
        model_path_to_load, target_speaker_id = None, 0
        direction = inf_conf.get('conversion_direction', 'specific')
        if direction == 'male_to_female': model_path_to_load,target_speaker_id = inf_conf.get('fine_tuned_model_path_female_generator'),inf_conf.get('target_female_speaker_id',0)
        elif direction == 'female_to_male': model_path_to_load,target_speaker_id = inf_conf.get('fine_tuned_model_path_male_generator'),inf_conf.get('target_male_speaker_id',1)
        elif direction == 'specific': model_path_to_load,target_speaker_id = inf_conf.get('specific_generator_checkpoint_path'),inf_conf.get('specific_target_speaker_id',0)
        else: logger.error(f"Неверное 'conversion_direction': {direction}. Выход."); sys.exit(1)
        if not model_path_to_load or not os.path.exists(model_path_to_load): logger.error(f"Модель для инференса не найдена: {model_path_to_load}. Выход."); sys.exit(1)
        convert_voice_from_file(config, model_path_to_load, inf_conf['input_wav_path'], inf_conf['output_wav_path'], target_speaker_id)


    elif effective_mode == 'inference_realtime':
        # ... (логика inference_realtime как в предыдущем шаге) ...
        logger.info("Настройка для режима 'inference_realtime'...")
        rt_conf = config.get('inference_realtime');
        if not rt_conf: logger.error("Секция `inference_realtime` отсутствует. Выход."); sys.exit(1)
        list_audio_devices()
        input_dev_idx = rt_conf.get('input_device_index', -1)
        output_dev_idx = rt_conf.get('output_device_index', -1)
        if input_dev_idx == -1 : input_dev_idx = select_device_id("Выберите ID устройства ВВОДА: ", kind="ввода"); rt_conf['input_device_index'] = input_dev_idx
        if output_dev_idx == -1 : output_dev_idx = select_device_id("Выберите ID устройства ВЫВОДА: ", kind="вывода"); rt_conf['output_device_index'] = output_dev_idx
        if input_dev_idx is None or output_dev_idx is None: logger.error("Выбор устройства отменен. Выход."); sys.exit(1)
        try:
            converter = RealTimeVoiceConverter(config); converter.start()
        except KeyboardInterrupt: logger.info("Преобразование остановлено пользователем.")
        except Exception as e: logger.error(f"Ошибка инференса в реальном времени: {e}", exc_info=True)
        finally:
            if 'converter' in locals() and hasattr(converter, 'stop'): converter.stop()
        logger.info("Сессия инференса в реальном времени завершена.")

    else:
        logger.error(f"Неизвестный режим: '{effective_mode}'. Выход.")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VoiceClonerPy: Система клонирования голоса.")
    parser.add_argument('--config', type=str, default="config/config_template.yaml", help="Путь к YAML файлу конфигурации.")
    parser.add_argument('--mode', type=str, choices=['train', 'fine_tune', 'inference_file', 'inference_realtime'], help="Переопределить режим работы.")
    cmd_args = parser.parse_args()
    main(cmd_args)

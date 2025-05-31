import os
import shutil
import subprocess
import csv
import random
import yaml
import soundfile as sf
import librosa
import numpy as np
import sys # Для sys.executable
import glob # Для find_latest_generator_checkpoint
import re   # Для find_latest_generator_checkpoint

# --- Глобальные Константы и Настройки для Теста ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TEST_VOICES_DIR = os.path.join(BASE_DIR, 'data', 'test_pipeline_voices')
MALE_VOICE_NAME = "male.wav"
FEMALE_VOICE_NAME = "female.wav"
MALE_VOICE_PATH = os.path.join(TEST_VOICES_DIR, MALE_VOICE_NAME)
FEMALE_VOICE_PATH = os.path.join(TEST_VOICES_DIR, FEMALE_VOICE_NAME)

PROCESSED_DATA_ROOT = os.path.join(BASE_DIR, 'data', 'pipeline_test_processed')
SPEAKER_MALE_DIR = os.path.join(PROCESSED_DATA_ROOT, 'speaker_male')
SPEAKER_FEMALE_DIR = os.path.join(PROCESSED_DATA_ROOT, 'speaker_female')
METADATA_DIR = os.path.join(PROCESSED_DATA_ROOT, 'metadata')
TRAIN_METADATA_FILE = os.path.join(METADATA_DIR, 'train_metadata.csv')
VAL_METADATA_FILE = os.path.join(METADATA_DIR, 'val_metadata.csv')

TEST_CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'pipeline_test_config.yaml')
CONFIG_TEMPLATE_PATH = os.path.join(BASE_DIR, 'config', 'config_template.yaml')

# Директория, куда Trainer будет сохранять чекпоинты (относительно BASE_DIR)
# Trainer сохраняет в os.path.join(config['training']['checkpoint_dir'], config['project']['experiment_name'])
# config['training']['checkpoint_dir'] будет установлен в 'checkpoints/' в generate_test_config
CHECKPOINTS_BASE_DIR_FROM_ROOT = 'checkpoints' # Эта директория будет в корне проекта

TARGET_SAMPLE_RATE = 22050
SEGMENT_DURATION_S = 2.5
TRAIN_VAL_SPLIT_RATIO = 0.9

TEST_TRAIN_EPOCHS = 1 # Минимальное количество эпох для создания чекпоинта
TEST_TRAIN_BATCH_SIZE = 2
TEST_VAL_BATCH_SIZE = 2
TEST_EXPERIMENT_NAME = "pipeline_test_experiment"

MALE_SPEAKER_ID = 0
FEMALE_SPEAKER_ID = 1

# Для инференса
INFERENCE_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "pipeline_test_output")


def create_dummy_wav_if_not_exists(path, duration_s=5, sr=TARGET_SAMPLE_RATE):
    if not os.path.exists(path):
        print(f"Файл {path} не найден, создаем фиктивный WAV...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        freq = 440; num_samples = int(duration_s * sr)
        time_axis = np.linspace(0, duration_s, num_samples, endpoint=False)
        dummy_audio = 0.3 * np.sin(2 * np.pi * freq * time_axis)
        sf.write(path, dummy_audio, sr)
        print(f"Фиктивный WAV файл {path} создан.")
    else: print(f"Используется существующий файл: {path}")

def segment_audio(input_wav_path, speaker_output_dir, segment_duration_s, target_sr):
    print(f"Сегментация аудио: {input_wav_path} в {speaker_output_dir}")
    if not os.path.exists(input_wav_path): print(f"Ошибка: Исходный WAV файл не найден: {input_wav_path}"); return []
    os.makedirs(speaker_output_dir, exist_ok=True)
    try: audio, sr = librosa.load(input_wav_path, sr=target_sr, mono=True)
    except Exception as e: print(f"Ошибка загрузки аудио {input_wav_path}: {e}"); return []
    total_duration_s = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(np.floor(total_duration_s / segment_duration_s)); segment_paths = []
    if num_segments == 0: print(f"Предупреждение: Аудио {input_wav_path} слишком короткое ({total_duration_s:.2f}с) для сегментов по {segment_duration_s}с."); return []
    for i in range(num_segments):
        start_sample, end_sample = int(i*segment_duration_s*sr), int((i+1)*segment_duration_s*sr)
        segment = audio[start_sample:end_sample]
        segment_path = os.path.join(speaker_output_dir, f"segment_{i+1}.wav")
        try: sf.write(segment_path, segment, sr); segment_paths.append(os.path.abspath(segment_path))
        except Exception as e: print(f"Ошибка сохранения сегмента {segment_path}: {e}")
    print(f"Создано {len(segment_paths)} сегментов из {input_wav_path}")
    return segment_paths

def create_metadata_file(segments_info_list, output_csv_path):
    print(f"Создание файла метаданных: {output_csv_path}")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['wav_path', 'speaker_id', 'speaker_name'])
        for info in segments_info_list: writer.writerow(info)
    print(f"Файл метаданных {output_csv_path} создан с {len(segments_info_list)} записями.")

def generate_test_config(template_path, output_config_path, train_meta_path, val_meta_path, processed_data_root):
    print(f"Генерация тестового конфигурационного файла: {output_config_path}")
    if not os.path.exists(template_path): print(f"Ошибка: Шаблон конфигурации не найден: {template_path}"); return None
    with open(template_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)

    config['project']['experiment_name'] = TEST_EXPERIMENT_NAME
    config['data']['base_data_path'] = os.path.abspath(processed_data_root)
    config['data']['train_metadata_file'] = os.path.abspath(train_meta_path)
    if val_meta_path: config['data']['val_metadata_file'] = os.path.abspath(val_meta_path)
    else: config['data']['val_metadata_file'] = None # Явно указываем None, если нет валидационных данных
    config['data']['sample_rate'] = TARGET_SAMPLE_RATE
    config['data']['batch_size'] = TEST_TRAIN_BATCH_SIZE
    config['data']['base_model_corpus_dir'] = None; config['data']['fine_tune_user_data_dir'] = None; config['data']['val_data_dir'] = None
    config['model']['num_speakers'] = 2
    config['training']['num_epochs'] = TEST_TRAIN_EPOCHS
    config['training']['batch_size_val'] = TEST_VAL_BATCH_SIZE
    # Путь к логам относительно корня проекта
    config['training']['log_file_path'] = os.path.join("logs", f"{TEST_EXPERIMENT_NAME}.log")
    # Базовая директория для чекпоинтов относительно корня проекта
    config['training']['checkpoint_dir'] = CHECKPOINTS_BASE_DIR_FROM_ROOT
    config['model']['vocoder']['checkpoint_path'] = "dummy_vocoder_checkpoint.pth"
    config['model']['vocoder']['config_path'] = "dummy_vocoder_config.json"
    # Убедимся, что пути абсолютные или корректно относительные для run.py
    config['training']['log_file_path'] = os.path.join(BASE_DIR, config['training']['log_file_path'])
    # config['training']['checkpoint_dir'] уже будет BASE_DIR/checkpoints/ в Trainer

    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Тестовый конфигурационный файл сохранен: {output_config_path}")
    return config # Возвращаем загруженный и измененный словарь конфигурации

def update_config_for_inference(config_dict, inference_input_wav, inference_output_wav,
                                model_checkpoint_path, target_conversion_direction, target_speaker_id_int):
    print(f"Обновление конфигурации для инференса: вход={inference_input_wav}, выход={inference_output_wav}")
    config_dict['training']['mode'] = 'inference_file' # Устанавливаем режим инференса

    # Очищаем параметры, не нужные для инференса, чтобы избежать путаницы
    config_dict['training']['resume_from_epoch'] = 0
    # ... другие параметры обучения можно сбросить или оставить ...

    inf_conf = config_dict.setdefault('inference_file', {})
    inf_conf['input_wav_path'] = inference_input_wav
    inf_conf['output_wav_path'] = inference_output_wav
    inf_conf['conversion_direction'] = target_conversion_direction

    # В зависимости от направления, устанавливаем путь к модели и ID целевого диктора
    if target_conversion_direction == 'male_to_female':
        inf_conf['fine_tuned_model_path_female_generator'] = model_checkpoint_path
        inf_conf['target_female_speaker_id'] = target_speaker_id_int
        # Очищаем другие пути, чтобы избежать неоднозначности
        inf_conf['fine_tuned_model_path_male_generator'] = None
        inf_conf['specific_generator_checkpoint_path'] = None
    elif target_conversion_direction == 'female_to_male':
        inf_conf['fine_tuned_model_path_male_generator'] = model_checkpoint_path
        inf_conf['target_male_speaker_id'] = target_speaker_id_int
        inf_conf['fine_tuned_model_path_female_generator'] = None
        inf_conf['specific_generator_checkpoint_path'] = None
    elif target_conversion_direction == 'specific': # Общий случай
        inf_conf['specific_generator_checkpoint_path'] = model_checkpoint_path
        inf_conf['specific_target_speaker_id'] = target_speaker_id_int
        inf_conf['fine_tuned_model_path_female_generator'] = None
        inf_conf['fine_tuned_model_path_male_generator'] = None
    else:
        print(f"Предупреждение: Неизвестное направление конверсии '{target_conversion_direction}' при обновлении конфига для инференса.")

    print(f"Конфигурация обновлена для инференса: направление={target_conversion_direction}, модель={model_checkpoint_path}, цель_ID={target_speaker_id_int}")
    return config_dict # Возвращаем измененный словарь

def find_latest_generator_checkpoint(checkpoints_root_dir, experiment_name, target_epoch=None):
    """
    Находит последний чекпоинт генератора для указанного эксперимента.
    Если target_epoch указан, ищет чекпоинт для этой эпохи.
    """
    experiment_checkpoint_dir = os.path.join(BASE_DIR, checkpoints_root_dir, experiment_name)
    print(f"Поиск чекпоинтов в: {experiment_checkpoint_dir}")
    if not os.path.isdir(experiment_checkpoint_dir):
        print(f"Предупреждение: Директория чекпоинтов эксперимента не найдена: {experiment_checkpoint_dir}")
        return None

    if target_epoch is not None: # Ищем чекпоинт для конкретной эпохи
        # Имена могут включать метрики, например: generator_epoch_X_train_g_loss_Y.pth
        # Или model_best_mcd_generator.pth
        # Для простоты ищем по 'epoch_X'
        search_pattern = os.path.join(experiment_checkpoint_dir, f"*generator_epoch_{target_epoch}*.pth")
    else: # Ищем самый последний по номеру эпохи
        search_pattern = os.path.join(experiment_checkpoint_dir, "*generator_epoch_*.pth")

    checkpoints = glob.glob(search_pattern)
    if not checkpoints:
        # Попробуем найти 'model_best_..._generator.pth' если обычные не найдены или target_epoch не указан
        if target_epoch is None: # Ищем лучший, только если не указана конкретная эпоха
            best_checkpoints = glob.glob(os.path.join(experiment_checkpoint_dir, "model_best_*_generator.pth"))
            if best_checkpoints:
                # Если есть несколько "лучших" (например, по разным метрикам), берем первый найденный
                print(f"Найден 'лучший' чекпоинт: {best_checkpoints[0]}")
                return best_checkpoints[0]

        print(f"Предупреждение: Чекпоинты генератора не найдены по шаблону: {search_pattern}")
        return None

    latest_checkpoint = None
    max_epoch = -1
    if target_epoch is None: # Если ищем самый последний по номеру
        for cp in checkpoints:
            filename = os.path.basename(cp)
            match = re.search(r"epoch_(\d+)", filename)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    latest_checkpoint = cp
        if latest_checkpoint: print(f"Найден последний чекпоинт (эпоха {max_epoch}): {latest_checkpoint}")
        else: print(f"Не удалось извлечь номер эпохи из найденных файлов: {checkpoints}")
    else: # Если искали конкретную эпоху, берем первый попавшийся (может быть несколько с метриками)
        latest_checkpoint = checkpoints[0]
        print(f"Найден чекпоинт для эпохи {target_epoch}: {latest_checkpoint}")

    return latest_checkpoint


def run_main_script(config_path, mode="train"):
    main_script_path = os.path.join(BASE_DIR, 'scripts', 'run.py')
    command = [sys.executable, main_script_path, '--config', config_path, '--mode', mode]
    print(f"Запуск команды: {' '.join(command)}")
    try:
        env = os.environ.copy()
        # BASE_DIR is the project root.
        # PYTHONPATH should point to BASE_DIR to allow finding modules like 'src'.
        # cwd should also be BASE_DIR.
        python_path_value = BASE_DIR + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = python_path_value

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', cwd=BASE_DIR, env=env)
        for line in process.stdout: print(line, end='')
        process.wait()
        if process.returncode != 0: print(f"Ошибка выполнения команды! Код возврата: {process.returncode}"); return False
        print(f"Команда '{mode}' успешно выполнена.")
        return True
    except Exception as e: print(f"Исключение при выполнении команды: {e}"); return False

if __name__ == '__main__':
    print("--- Начало Теста Процесса VoiceClonerPy ---")
    create_dummy_wav_if_not_exists(MALE_VOICE_PATH); create_dummy_wav_if_not_exists(FEMALE_VOICE_PATH)
    if os.path.exists(PROCESSED_DATA_ROOT): shutil.rmtree(PROCESSED_DATA_ROOT)
    os.makedirs(SPEAKER_MALE_DIR); os.makedirs(SPEAKER_FEMALE_DIR); os.makedirs(METADATA_DIR)
    os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True); os.makedirs(os.path.join(BASE_DIR, CHECKPOINTS_BASE_DIR_FROM_ROOT), exist_ok=True)
    os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

    male_segments = segment_audio(MALE_VOICE_PATH, SPEAKER_MALE_DIR, SEGMENT_DURATION_S, TARGET_SAMPLE_RATE)
    female_segments = segment_audio(FEMALE_VOICE_PATH, SPEAKER_FEMALE_DIR, SEGMENT_DURATION_S, TARGET_SAMPLE_RATE)
    if not male_segments or not female_segments: print("Ошибка: Не удалось создать аудио сегменты."); sys.exit(1)

    all_segments_info = [(p, MALE_SPEAKER_ID, "male_test") for p in male_segments] + \
                        [(p, FEMALE_SPEAKER_ID, "female_test") for p in female_segments]
    random.shuffle(all_segments_info)
    split_idx = int(len(all_segments_info) * TRAIN_VAL_SPLIT_RATIO)
    train_info, val_info = all_segments_info[:split_idx], all_segments_info[split_idx:]
    if not train_info: print("Ошибка: Нет данных для обучения."); sys.exit(1)

    create_metadata_file(train_info, TRAIN_METADATA_FILE)
    current_val_meta_file = VAL_METADATA_FILE
    if val_info: create_metadata_file(val_info, VAL_METADATA_FILE)
    else: print("Предупреждение: Нет данных для валидации."); current_val_meta_file = None

    # Генерация конфига для обучения
    loaded_train_config = generate_test_config(CONFIG_TEMPLATE_PATH, TEST_CONFIG_PATH,
                                               TRAIN_METADATA_FILE, current_val_meta_file, PROCESSED_DATA_ROOT)
    if not loaded_train_config: print("Ошибка генерации конфига обучения."); sys.exit(1)

    print("\n--- Запуск этапа обучения (Train Mode) ---")
    train_success = run_main_script(TEST_CONFIG_PATH, mode="train")

    if train_success:
        print("\n--- Тестовое обучение завершено. Подготовка к тестовому инференсу... ---")

        # Ищем чекпоинт последней эпохи обучения
        # Trainer сохраняет чекпоинты в <checkpoint_dir>/<experiment_name>/
        # CHECKPOINTS_BASE_DIR_FROM_ROOT = 'checkpoints' (из config['training']['checkpoint_dir'])
        # TEST_EXPERIMENT_NAME = loaded_train_config['project']['experiment_name']
        # num_train_epochs = loaded_train_config['training']['num_epochs'] # Это 0-индексированные эпохи

        # find_latest_generator_checkpoint ожидает target_epoch как 1-индексированный номер эпохи
        latest_checkpoint_path = find_latest_generator_checkpoint(
            CHECKPOINTS_BASE_DIR_FROM_ROOT,
            TEST_EXPERIMENT_NAME,
            target_epoch=TEST_TRAIN_EPOCHS # Ищем чекпоинт последней обученной эпохи (1-индексированный)
        )

        if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
            print(f"Найден чекпоинт для инференса: {latest_checkpoint_path}")

            # Выбираем один из валидационных или тренировочных сегментов для инференса
            inference_input_wav_path = ""
            if val_info: inference_input_wav_path = val_info[0][0] # Берем первый валидационный файл
            elif train_info: inference_input_wav_path = train_info[0][0] # Или первый тренировочный
            else: print("Нет доступных аудиофайлов для инференса."); sys.exit(1)

            # Убедимся, что путь абсолютный, т.к. конфиг может быть в другом месте
            if not os.path.isabs(inference_input_wav_path):
                inference_input_wav_path = os.path.join(PROCESSED_DATA_ROOT, inference_input_wav_path) # Это неверно, т.к. segment_audio уже дает abs путь

            inference_output_wav = os.path.join(INFERENCE_OUTPUT_DIR, "converted_pipeline_test_voice.wav")

            # Пример: конвертация из мужского в женский (male_id=0 -> female_id=1)
            target_gender_conversion = 'male_to_female'
            target_inference_speaker_id = FEMALE_SPEAKER_ID

            # Обновляем тот же словарь конфига, который вернул generate_test_config
            inference_config_dict = update_config_for_inference(
                loaded_train_config,
                inference_input_wav_path,
                inference_output_wav,
                latest_checkpoint_path,
                target_gender_conversion,
                target_inference_speaker_id
            )

            # Сохраняем обновленный конфиг (перезаписываем тот же файл)
            with open(TEST_CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(inference_config_dict, f, sort_keys=False, allow_unicode=True)
            print(f"Конфигурационный файл обновлен для инференса: {TEST_CONFIG_PATH}")

            print(f"\n--- Запуск тестового инференса: {inference_input_wav_path} -> {target_gender_conversion} (ID: {target_inference_speaker_id}) ---")
            run_main_script(TEST_CONFIG_PATH, mode="inference_file")
            print(f"--- Тестовый инференс завершен. Результат (возможно) сохранен в: {inference_output_wav} ---")
        else:
            print(f"--- Чекпоинт генератора после обучения не найден (ожидался для эпохи {TEST_TRAIN_EPOCHS}). Пропуск тестового инференса. ---")
    else:
        print("\n--- Ошибка на этапе обучения. Инференс не будет запущен. ---")

    print("\n--- Тест Процесса VoiceClonerPy Завершен ---")

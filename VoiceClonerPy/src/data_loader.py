import os
import librosa
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .audio_utils import wav_to_mel_spectrogram
import logging # Для логгирования в этом модуле

logger = logging.getLogger(__name__)

def load_wav(file_path, sample_rate=None):
    # Загружает WAV файл.
    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        return None, None
    try:
        audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        return audio_data, sr
    except Exception as e:
        logger.error(f"Ошибка загрузки WAV файла {file_path}: {e}")
        return None, None

def save_wav(file_path, audio_data, sample_rate):
    # Сохраняет аудиоданные в WAV файл.
    try:
        sf.write(file_path, audio_data, sample_rate)
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения WAV файла {file_path}: {e}")
        return False

def prepare_fine_tune_data(user_data_dir):
    # Подготовка данных для fine-tuning (пока просто список WAV файлов).
    if not os.path.isdir(user_data_dir):
        logger.error(f"Директория не найдена: {user_data_dir}")
        return []
    try:
        wav_files = [os.path.join(user_data_dir, f) for f in os.listdir(user_data_dir) if f.lower().endswith('.wav')]
        logger.info(f"Найдено {len(wav_files)} WAV файлов в {user_data_dir} для fine-tuning.")
        return wav_files
    except Exception as e:
        logger.error(f"Ошибка доступа к директории {user_data_dir}: {e}")
        return []

def prepare_base_model_data(corpus_dir):
    # Подготовка данных для базовой модели (пока просто список WAV файлов).
    # В реальной реализации здесь может быть более сложная логика, включая метаданные.
    if not os.path.isdir(corpus_dir):
        logger.error(f"Директория корпуса не найдена: {corpus_dir}")
        return []
    try:
        # Эта функция должна возвращать список путей к файлам, а не просто имена файлов
        audio_files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if f.lower().endswith(('.wav', '.flac', '.mp3'))] # Пример с разными форматами
        logger.info(f"Найдено {len(audio_files)} аудиофайлов в {corpus_dir}.")
        return audio_files
    except Exception as e:
        logger.error(f"Ошибка доступа к директории корпуса {corpus_dir}: {e}")
        return []

# Добавлено в предыдущем шаге, используется в scripts/run.py
def load_data_for_dataset(data_dir_or_file_list, metadata_file, num_speakers_from_model, is_validation=False):
    # Плейсхолдер: Эта функция должна парсить metadata_file или сканировать data_dir_or_file_list
    # для создания списка кортежей: [(audio_path, speaker_id)] для обучения,
    # или [(src_path, target_path, src_id, target_id)] для валидации.
    # Пока возвращает заглушечные данные.
    logger.warning(f"Загрузка данных для {'валидации' if is_validation else 'обучения'} использует ЗАГЛУШЕЧНЫЕ данные. Реализуйте реальную загрузку данных.")
    num_dummy = 20 if is_validation else 100

    if is_validation: # Ожидаемая структура: (source_mel_audio_path, target_eval_audio_path, source_speaker_id, target_speaker_id_for_conversion)
        return [(f"dummy_val_src_{i}.wav", f"dummy_val_trg_{i}.wav", i % num_speakers_from_model, (i+1) % num_speakers_from_model) for i in range(num_dummy)]
    else: # Ожидаемая структура: (audio_path, speaker_id)
        return [(f"dummy_train_{i}.wav", i % num_speakers_from_model) for i in range(num_dummy)]


class VoiceDataset(Dataset):
    def __init__(self, data_entries, data_config, is_validation=False):
        # data_entries: список кортежей, структура зависит от is_validation
        # Для обучения: [(wav_path1, speaker_id1), ...]
        # Для валидации: [(source_mel_audio_path, target_eval_audio_path, source_speaker_id, target_speaker_id_for_conversion), ...]
        self.data_entries = data_entries
        self.data_config = data_config
        self.is_validation = is_validation

        self.sample_rate = data_config['sample_rate']
        self.n_fft = data_config['n_fft']; self.hop_length = data_config['hop_length']; self.n_mels = data_config['n_mels']
        self.fmin = data_config.get('fmin',0); self.fmax = data_config.get('fmax', self.sample_rate/2); self.power_to_db = data_config.get('power_to_db',True)

        logger.info(f"VoiceDataset инициализирован с {len(self.data_entries)} сэмплами. Режим валидации: {self.is_validation}")

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]

        if self.is_validation:
            # Запись для валидации: (source_mel_audio_path, target_eval_audio_path, source_speaker_id, target_speaker_id_for_conversion)
            source_audio_path, target_eval_audio_path, source_speaker_id, target_speaker_id_for_conversion = entry

            source_audio_data, _ = load_wav(source_audio_path, sample_rate=self.sample_rate)
            if source_audio_data is None:
                logger.warning(f"Ошибка загрузки исходного аудио для валидации: {source_audio_path}. Возвращаем заглушку.")
                # Возвращаем заглушки, соответствующие ожидаемой структуре Trainer.evaluate_epoch
                return torch.zeros((self.n_mels, 128)), torch.tensor(source_speaker_id, dtype=torch.long), \
                       torch.tensor(target_speaker_id_for_conversion, dtype=torch.long), "dummy_path_error.wav"

            source_mel_np = wav_to_mel_spectrogram(source_audio_data, self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                                                   n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power_to_db=self.power_to_db)
            source_mel_tensor = torch.from_numpy(source_mel_np).float()

            return source_mel_tensor, torch.tensor(source_speaker_id, dtype=torch.long), \
                   torch.tensor(target_speaker_id_for_conversion, dtype=torch.long), target_eval_audio_path
        else: # Режим обучения
            audio_path, speaker_id = entry
            audio_data, _ = load_wav(audio_path, sample_rate=self.sample_rate)
            if audio_data is None:
                logger.warning(f"Ошибка загрузки аудио для обучения: {audio_path}. Возвращаем заглушку.")
                return torch.zeros((self.n_mels, 128)), torch.tensor(speaker_id, dtype=torch.long)

            mel_spectrogram_np = wav_to_mel_spectrogram(audio_data, self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                                                        n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power_to_db=self.power_to_db)
            mel_tensor = torch.from_numpy(mel_spectrogram_np).float()
            return mel_tensor, torch.tensor(speaker_id, dtype=torch.long)

import os
import librosa
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .audio_utils import wav_to_mel_spectrogram
import logging

logger = logging.getLogger(__name__)

def load_wav(file_path, sample_rate=None):
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
    try:
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            logger.info(f"Создана директория для сохранения: {parent_dir}")

        # Обеспечиваем C-contiguous формат
        audio_data = np.ascontiguousarray(audio_data)

        sf.write(file_path, audio_data, sample_rate, format='WAV', subtype='FLOAT')
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения WAV файла {file_path}: {e}")
        return False

# Функции prepare_fine_tune_data и prepare_base_model_data остаются, но могут быть заменены на load_data_for_dataset
def prepare_fine_tune_data(user_data_dir):
    if not os.path.isdir(user_data_dir): logger.error(f"Директория не найдена: {user_data_dir}"); return []
    try:
        wav_files = [os.path.join(user_data_dir, f) for f in os.listdir(user_data_dir) if f.lower().endswith('.wav')]
        logger.info(f"Найдено {len(wav_files)} WAV файлов в {user_data_dir} для fine-tuning.")
        return wav_files
    except Exception as e: logger.error(f"Ошибка доступа к {user_data_dir}: {e}"); return []

def prepare_base_model_data(corpus_dir):
    if not os.path.isdir(corpus_dir): logger.error(f"Директория корпуса не найдена: {corpus_dir}"); return []
    try:
        audio_files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if f.lower().endswith(('.wav', '.flac', '.mp3'))]
        logger.info(f"Найдено {len(audio_files)} аудиофайлов в {corpus_dir}.")
        return audio_files
    except Exception as e: logger.error(f"Ошибка доступа к {corpus_dir}: {e}"); return []

# Эта функция теперь в scripts/run.py, но может быть и здесь, если импортировать csv
# def load_data_for_dataset(...): ...

class VoiceDataset(Dataset):
    def __init__(self, data_config, data_entries, is_validation=False):
        # data_entries: список кортежей [(wav_path1, speaker_id1), ...]
        # Для валидации, если is_validation=True, data_entries может содержать
        # (source_wav_path, target_original_wav_path, source_speaker_id, target_conversion_speaker_id)
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
        global source_speaker_id, target_speaker_id_for_conversion, target_original_audio_path, speaker_id
        entry = self.data_entries[idx]

        if self.is_validation:
            # Ожидаемая структура entry для валидации:
            # (source_audio_path, target_original_audio_path, source_speaker_id, target_speaker_id_for_conversion)
            # source_audio_path: аудио для извлечения мел-спектрограммы (вход генератора)
            # target_original_audio_path: путь к оригинальному WAV файлу целевого диктора (для объективной оценки)
            # source_speaker_id: ID исходного диктора (не используется в текущей логике Trainer.evaluate_epoch, но может быть полезен)
            # target_speaker_id_for_conversion: ID целевого диктора для генератора
            source_audio_path, target_original_audio_path, source_speaker_id, target_speaker_id_for_conversion = entry

            audio_to_process_for_mel = source_audio_path # Аудио, из которого делаем мел для входа в генератор
        else: # Режим обучения
            # Ожидаемая структура entry для обучения: (audio_path, speaker_id)
            audio_to_process_for_mel, speaker_id = entry
            # В режиме обучения target_original_audio_path не нужен напрямую для Trainer,
            # но если бы мы хотели его вернуть, он был бы равен audio_path.

        # Загрузка и обработка аудио для мел-спектрограммы
        audio_data, _ = load_wav(audio_to_process_for_mel, sample_rate=self.sample_rate)
        if audio_data is None:
            err_msg = f"Ошибка загрузки аудио: {audio_to_process_for_mel}. Возвращаем заглушку."
            logger.warning(err_msg)
            dummy_mel = torch.zeros((self.n_mels, 128)) # Форма заглушки
            if self.is_validation:
                # source_mel, source_id, target_id_for_conversion, original_wav_path_for_eval
                return dummy_mel, torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long), "error_path.wav"
            else:
                return dummy_mel, torch.tensor(0, dtype=torch.long)

        mel_spectrogram_np = wav_to_mel_spectrogram(audio_data, self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                                                   n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power_to_db=self.power_to_db)
        mel_tensor = torch.from_numpy(mel_spectrogram_np).float()

        if self.is_validation:
            # Возвращаем: source_mel_tensor, source_speaker_id, target_speaker_id_for_conversion, target_original_audio_path
            # Trainer.evaluate_epoch использует source_mel_tensor и target_speaker_id_for_conversion для генератора,
            # и target_original_audio_path для загрузки эталонного аудио для метрик.
            return mel_tensor, torch.tensor(source_speaker_id, dtype=torch.long), \
                   torch.tensor(target_speaker_id_for_conversion, dtype=torch.long), target_original_audio_path
        else: # Режим обучения
            # Возвращаем: mel_tensor, speaker_id (этот speaker_id будет и source, и target для некоторых потерь в StarGAN-VC)
            return mel_tensor, torch.tensor(speaker_id, dtype=torch.long)

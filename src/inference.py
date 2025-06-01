import torch
import logging
import os
import numpy as np

from .models.stargan_vc import Generator
from .models.vocoder import HiFiGANVocoder
from .data_loader import load_wav, save_wav
from .audio_utils import wav_to_mel_spectrogram

# Настройка базового логгера для инференса
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def convert_voice_from_file(full_config, generator_model_path, input_wav_path, output_wav_path, target_speaker_id_for_generator):
    """
    Конвертирует голос из входного WAV файла в голос целевого диктора.
    """
    logger.info(f"Начало конвертации голоса для: {input_wav_path}")
    logger.info(f"Путь к модели генератора: {generator_model_path}")
    logger.info(f"ID целевого диктора для генератора: {target_speaker_id_for_generator}")

    try:
        # Определение устройства
        if full_config['training']['device'] == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        logger.info(f"Используется устройство: {device}")

        # Инициализация Генератора
        generator = Generator(config=full_config['model']).to(device)

        logger.info(f"Загрузка состояния генератора из: {generator_model_path}")
        if not os.path.exists(generator_model_path):
            logger.error(f"Чекпоинт генератора не найден: {generator_model_path}")
            return False

        checkpoint = torch.load(generator_model_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        elif 'model_state_dict' in checkpoint and 'generator' in checkpoint['model_state_dict']:
             generator.load_state_dict(checkpoint['model_state_dict']['generator'])
        elif 'state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['state_dict'])
        else:
            generator.load_state_dict(checkpoint)

        generator.eval()
        logger.info("Модель генератора загружена и переведена в режим оценки (eval).")

        # Инициализация Вокодера
        vocoder_checkpoint_path = full_config['model']['vocoder']['checkpoint_path']
        if not vocoder_checkpoint_path or not os.path.exists(vocoder_checkpoint_path):
            logger.warning(f"Путь к чекпоинту вокодера не найден или не указан: {vocoder_checkpoint_path}. Вокодер может возвращать 'dummy' аудио.")
        vocoder = HiFiGANVocoder(vocoder_checkpoint_path=vocoder_checkpoint_path, main_app_config=full_config)

        # Загрузка исходного аудио
        logger.info(f"Загрузка исходного аудио: {input_wav_path}")
        audio_data, sr = load_wav(input_wav_path, sample_rate=full_config['data']['sample_rate'])
        if audio_data is None:
            logger.error(f"Не удалось загрузить аудио из {input_wav_path}")
            return False
        logger.info(f"Аудио загружено. Частота дискретизации: {sr}, Длительность: {len(audio_data)/sr:.2f}с")

        # Конвертация аудио в мел-спектрограмму
        logger.info("Конвертация аудио в мел-спектрограмму...")
        mel_spec = wav_to_mel_spectrogram(
            audio_data, sample_rate=sr,
            n_fft=full_config['data']['n_fft'], hop_length=full_config['data']['hop_length'],
            n_mels=full_config['data']['n_mels'], fmin=full_config['data']['fmin'],
            fmax=full_config['data']['fmax'], power_to_db=full_config['data']['power_to_db']
        )
        logger.info(f"Мел-спектрограмма создана. Форма: {mel_spec.shape}")

        # Подготовка тензора мел-спектрограммы для генератора
        input_mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).to(device)

        # Подготовка тензора эмбеддинга целевого диктора
        num_speakers = full_config['model']['num_speakers']
        speaker_embedding_dim = full_config['model']['speaker_embedding_dim']
        # Временный слой эмбеддингов для инференса; в идеале - часть общей системы управления эмбеддингами
        temp_speaker_embedding_layer = nn.Embedding(num_speakers, speaker_embedding_dim).to(device)
        target_speaker_emb_tensor = temp_speaker_embedding_layer(
            torch.tensor([target_speaker_id_for_generator], dtype=torch.long).to(device)
        )
        logger.info(f"Тензор эмбеддинга целевого диктора создан. Форма: {target_speaker_emb_tensor.shape}")

        # Выполнение инференса
        logger.info("Выполнение преобразования голоса (инференс генератора)...")
        with torch.no_grad():
            output_mel_tensor = generator(input_mel_tensor, target_speaker_emb_tensor)
        logger.info(f"Выходная мел-спектрограмма от генератора. Форма: {output_mel_tensor.shape}")

        # Конвертация выходной мел-спектрограммы в аудио с помощью Вокодера
        logger.info("Конвертация выходной мел-спектрограммы в аудио (инференс вокодера)...")
        if output_mel_tensor.ndim == 3 and output_mel_tensor.size(0) == 1:
             processed_output_mel = output_mel_tensor.squeeze(0)
        else: processed_output_mel = output_mel_tensor
        output_waveform = vocoder.mel_to_wav(processed_output_mel.cpu()) # Вокодер может ожидать тензор на CPU
        logger.info(f"Выходное аудио сгенерировано. Форма: {output_waveform.shape}, Тип: {output_waveform.dtype}")

        output_waveform_np = output_waveform.detach().cpu().numpy() if isinstance(output_waveform, torch.Tensor) else output_waveform

        # Проверка и очистка от NaN/Inf
        if np.isnan(output_waveform_np).any() or np.isinf(output_waveform_np).any():
            logger.warning("Обнаружены NaN или Inf значения в выходных аудиоданных. Производится очистка...")
            output_waveform_np = np.nan_to_num(output_waveform_np, nan=0.0, posinf=0.0, neginf=0.0) # Заменяем на 0.0
            logger.info("NaN/Inf значения заменены на 0.")

        # Нормализация аудиоданных
        if output_waveform_np.size > 0:
            max_val = np.max(np.abs(output_waveform_np))
            if max_val > 1.0:
                output_waveform_np = output_waveform_np / max_val
                logger.info(f"Аудиоданные нормализованы (макс. абс. значение было {max_val:.4f}).")
            elif max_val == 0: # Избегаем деления на ноль если весь сигнал нулевой
                logger.warning("Аудиоданные состоят из нулей, нормализация не требуется/невозможна.")
            else: # max_val <= 1.0 and max_val > 0
                logger.info(f"Аудиоданные уже в диапазоне [-1.0, 1.0] (макс. абс. значение {max_val:.4f}), дополнительная нормализация не применена.")
        else:
            logger.warning("Аудиоданные пусты, нормализация невозможна.")

        # Сохранение выходного аудио
        logger.info(f"Сохранение выходного аудио в: {output_wav_path}")
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True) # Создаем директорию, если ее нет
        success = save_wav(output_wav_path, output_waveform_np, full_config['data']['sample_rate'])
        if success: logger.info("Преобразование голоса успешно завершено.")
        else: logger.error("Не удалось сохранить выходное аудио.")
        return success
    except Exception as e:
        logger.error(f"Произошла ошибка во время преобразования голоса: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    # Пример использования (требует наличия конфигурационного файла и моделей)
    print("Этот скрипт предназначен для вызова из run.py или программно.")

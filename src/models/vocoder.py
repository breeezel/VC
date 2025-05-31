import torch
import torch.nn as nn
import numpy as np
import os # Для проверки существования файла чекпоинта
import logging # Для логирования

# Импорт компонентов HiFi-GAN и AttrDict из локального файла
from .hifi_gan_components import Generator as HiFiGANGeneratorModel, AttrDict

logger = logging.getLogger(__name__)

class HiFiGANVocoder(nn.Module):
    def __init__(self, vocoder_checkpoint_path, main_app_config):
        super(HiFiGANVocoder, self).__init__()
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.main_app_config = main_app_config # Сохраняем основную конфигурацию приложения

        self.device = torch.device(main_app_config['training']['device'] if torch.cuda.is_available() and main_app_config['training']['device'] == 'cuda' else 'cpu')

        # TODO: Загрузить специфичную конфигурацию HiFi-GAN, если она отделена от чекпоинта
        #       или передается через main_app_config['model']['vocoder']['hifi_gan_config']
        # Пока что создаем dummy hifigan_model_config на основе общих значений по умолчанию для HiFi-GAN V1
        hifigan_model_config = AttrDict({
            "num_mels": main_app_config['data']['n_mels'], # Важно, чтобы совпадало с данными
            "resblock": "1", # Тип ResBlock (1 или 2)
            "upsample_rates": [8, 8, 2, 2], # Коэффициенты повышения дискретизации
            "upsample_kernel_sizes": [16, 16, 4, 4], # Размеры ядер для ConvTranspose1d
            "upsample_initial_channel": 512, # Начальное количество каналов перед повышением дискретизации
            "resblock_kernel_sizes": [3, 7, 11], # Размеры ядер в ResBlock'ах
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], # Дилатации в ResBlock'ах
            # Другие параметры, если модель hifi_gan_components.Generator их ожидает
        })

        self.model = HiFiGANGeneratorModel(h=hifigan_model_config).to(self.device)

        if vocoder_checkpoint_path and os.path.exists(vocoder_checkpoint_path):
            try:
                logger.info(f"Загрузка чекпоинта HiFi-GAN из: {vocoder_checkpoint_path}")
                checkpoint_dict = torch.load(vocoder_checkpoint_path, map_location=self.device)

                # Ключи в словаре чекпоинта могут отличаться
                if 'generator' in checkpoint_dict:
                    self.model.load_state_dict(checkpoint_dict['generator'])
                elif 'state_dict' in checkpoint_dict: # Иногда сохраняют всю модель или state_dict под этим ключом
                     self.model.load_state_dict(checkpoint_dict['state_dict'])
                else: # Если чекпоинт - это просто state_dict генератора
                    self.model.load_state_dict(checkpoint_dict)

                self.model.eval()
                self.model.remove_weight_norm() # Обычно выполняется после обучения для HiFi-GAN
                logger.info(f"Реальная модель HiFi-GAN успешно загружена из {vocoder_checkpoint_path} и переведена в режим оценки.")
            except Exception as e:
                logger.error(f"Ошибка загрузки чекпоинта HiFi-GAN: {e}. Вокодер будет работать в режиме заглушки.", exc_info=True)
                self.model = None # Возврат к заглушке при ошибке загрузки
        else:
            logger.warning(f"Не удалось загрузить чекпоинт HiFi-GAN (путь: '{vocoder_checkpoint_path}' не существует или не указан). Вокодер будет работать в режиме заглушки (синусоида).")
            self.model = None

        if self.model is None: # Дополнительное сообщение, если модель осталась None
            logger.warning("Инициализация HiFiGANVocoder: self.model is None. mel_to_wav будет генерировать синусоиду.")


    @torch.no_grad()
    def mel_to_wav(self, mel_spectrogram):
        if self.model is None:
            logger.warning("HiFi-GAN: Модель не загружена. Возвращаем 'dummy' синусоиду.") # Сообщение на русском
            if not isinstance(mel_spectrogram, torch.Tensor): mel_spectrogram = torch.tensor(mel_spectrogram).float() # Убедимся, что это тензор
            mel_spectrogram = mel_spectrogram.to(self.device) # Перемещаем на устройство для一致性

            if mel_spectrogram.ndim == 3: num_frames, batch_size = mel_spectrogram.shape[2], mel_spectrogram.shape[0]
            elif mel_spectrogram.ndim == 2: num_frames, batch_size = mel_spectrogram.shape[1], 1
            else: raise ValueError(f"Мел-спектрограмма имеет неожиданное кол-во измерений: {mel_spectrogram.ndim}")

            sample_rate = self.main_app_config['data']['sample_rate']
            hop_length = self.main_app_config['data']['hop_length']
            expected_samples = num_frames * hop_length

            freq = 440.0; time_axis = torch.linspace(0, expected_samples / sample_rate, steps=expected_samples, device=self.device)
            dummy_waveform = 0.5 * torch.sin(2 * np.pi * freq * time_axis)

            output = dummy_waveform.unsqueeze(0).repeat(batch_size, 1) if batch_size > 1 and mel_spectrogram.ndim == 3 else dummy_waveform
            return output.cpu() # Возвращаем на CPU, как ожидается от функции

        # Реальный инференс HiFi-GAN
        # Убедимся, что mel_spectrogram на правильном устройстве и имеет правильную форму (B, C, T)
        if not isinstance(mel_spectrogram, torch.Tensor): mel_spectrogram = torch.tensor(mel_spectrogram).float()
        mel_spectrogram = mel_spectrogram.to(self.device)
        if mel_spectrogram.ndim == 2: mel_spectrogram = mel_spectrogram.unsqueeze(0) # (C, T) -> (1, C, T)

        output_waveform = self.model(mel_spectrogram) # (B, 1, Samples)
        # .squeeze(1) удаляет канал, .cpu() перемещает на CPU. Если батч=1, можно .squeeze(0) для (Samples,)
        return output_waveform.squeeze(1).cpu()

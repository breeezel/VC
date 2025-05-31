import torch
import sounddevice as sd
import numpy as np
import logging
import time
import collections
import os

from .models.stargan_vc import Generator
from .models.vocoder import HiFiGANVocoder
from .audio_utils import wav_to_mel_spectrogram

try:
    import noisereduce
except ImportError:
    noisereduce = None

logger = logging.getLogger(__name__)
# Уровни и обработчики настраиваются в run.py или при инициализации этого класса, если он используется отдельно
if not logger.handlers: # Предотвращаем дублирование обработчиков
    ch = logging.StreamHandler(); formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter); logger.addHandler(ch)
if noisereduce is None: logger.warning("Библиотека 'noisereduce' не найдена. Подавление шума будет недоступно.")


class RealTimeVoiceConverter:
    # Класс для преобразования голоса в реальном времени
    def __init__(self, full_config):
        self.config = full_config
        self.rt_config = full_config['inference_realtime']
        self.data_config = full_config['data']
        self.model_config = full_config['model']

        self.device = torch.device(full_config['training']['device'] if torch.cuda.is_available() and full_config['training']['device'] == 'cuda' else 'cpu')
        logger.info(f"RealTimeVoiceConverter использует устройство: {self.device}")

        # Параметры обработки аудио из конфигурации
        self.sample_rate = self.data_config['sample_rate']
        self.n_fft = self.data_config['n_fft']
        self.hop_length = self.data_config['hop_length']
        self.n_mels = self.data_config['n_mels']
        self.fmin = self.data_config.get('fmin', 0)
        self.fmax = self.data_config.get('fmax', self.sample_rate / 2)
        self.power_to_db = self.data_config.get('power_to_db', True)

        # Размер блока для аудиопотока (от sounddevice)
        self.stream_block_size = self.rt_config.get('buffer_size_samples', int(0.1 * self.sample_rate))

        # Инициализация буферов
        self.input_buffer = np.array([], dtype=np.float32) # Буфер для необработанного входного аудио
        self.output_audio_buffer = collections.deque()    # Очередь для обработанных аудио чанков на воспроизведение
        self.processed_audio_for_output = np.array([], dtype=np.float32) # Буфер для сборки данных для outdata

        logger.info("Загрузка модели Генератора для реального времени...")
        self.generator = Generator(config=self.model_config).to(self.device)

        gen_path_key_map = {"male_to_female": "fine_tuned_model_path_female_generator", "female_to_male": "fine_tuned_model_path_male_generator", "specific": "specific_generator_checkpoint_path"}
        direction = self.rt_config.get('conversion_direction', 'specific')
        gen_model_path_key = gen_path_key_map.get(direction, 'specific_generator_checkpoint_path')
        generator_checkpoint_path = self.rt_config.get(gen_model_path_key) # Не используем fallback на model_config здесь, путь должен быть явным для RT

        if generator_checkpoint_path and os.path.exists(generator_checkpoint_path):
            checkpoint_g = torch.load(generator_checkpoint_path, map_location=self.device)
            # Обработка различных форматов чекпоинтов
            if 'generator_state_dict' in checkpoint_g: self.generator.load_state_dict(checkpoint_g['generator_state_dict'])
            elif 'state_dict' in checkpoint_g: self.generator.load_state_dict(checkpoint_g['state_dict'])
            else: self.generator.load_state_dict(checkpoint_g)
            logger.info(f"Генератор загружен из {generator_checkpoint_path}")
        else:
            msg = f"Чекпоинт генератора не найден или не указан: {generator_checkpoint_path}. Конвертация в реальном времени не будет работать."
            logger.error(msg); raise FileNotFoundError(msg)
        self.generator.eval()

        logger.info("Загрузка модели Вокодера HiFi-GAN...")
        vocoder_checkpoint = self.model_config['vocoder']['checkpoint_path']
        self.vocoder = HiFiGANVocoder(checkpoint_path=vocoder_checkpoint, main_app_config=full_config) # Передаем main_app_config
        if self.vocoder.model is None: logger.warning("Модель HiFi-GAN в вокодере не загружена (None). Воспроизведение будет 'dummy' аудио.")

        # Подавление шума
        self.noise_reduction_enabled = self.rt_config.get('noise_suppression_enabled', False)
        self.noise_prop_decrease = self.rt_config.get('noise_prop_decrease', 1.0)
        if self.noise_reduction_enabled and noisereduce: logger.info(f"Подавление шума включено (prop_decrease={self.noise_prop_decrease}).")
        elif self.noise_reduction_enabled and not noisereduce: logger.warning("Подавление шума включено в конфиге, но библиотека 'noisereduce' не найдена.")
        else: logger.info("Подавление шума выключено.")

        # ID и эмбеддинг целевого диктора
        if direction == 'male_to_female': self.target_speaker_id = self.rt_config.get('target_female_speaker_id',0)
        elif direction == 'female_to_male': self.target_speaker_id = self.rt_config.get('target_male_speaker_id',1)
        else: self.target_speaker_id = self.rt_config.get('specific_target_speaker_id',0)
        logger.info(f"ID целевого диктора для конвертации: {self.target_speaker_id}")

        # Используем _get_speaker_embedding из Trainer или аналогичный механизм
        # Для простоты, создаем временный слой эмбеддингов здесь
        temp_speaker_embedding_layer = nn.Embedding(self.model_config['num_speakers'], self.model_config['speaker_embedding_dim']).to(self.device)
        self.target_speaker_embedding = temp_speaker_embedding_layer(torch.tensor([self.target_speaker_id], dtype=torch.long).to(self.device))
        logger.info(f"Эмбеддинг целевого диктора подготовлен. Форма: {self.target_speaker_embedding.shape}")

        self.stream = None # Атрибут для хранения аудиопотока

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        # indata: входной аудио чанк (numpy array [frames, channels])
        # outdata: буфер для выходного аудио чанка (numpy array [frames, channels])
        if status: logger.warning(f"Статус аудиопотока: {status}")

        current_chunk = indata[:, 0].astype(np.float32) # Используем только первый канал

        # 1. Подавление шума (если включено)
        if self.noise_reduction_enabled and noisereduce:
            try:
                current_chunk = noisereduce.reduce_noise(y=current_chunk, sr=self.sample_rate, prop_decrease=self.noise_prop_decrease, verbose=False)
            except Exception as e:
                logger.error(f"Ошибка при подавлении шума: {e}", exc_info=False) # Не выводим полный стектрейс в колбэке

        # 2. Добавляем новый чанк во входной буфер
        self.input_buffer = np.concatenate((self.input_buffer, current_chunk))

        # 3. Обработка, пока во входном буфере достаточно данных для n_fft (для первого окна)
        # или hop_length (для последующих окон в более сложной реализации с перекрытием)
        # Упрощенная логика: обрабатываем, если есть хотя бы n_fft сэмплов.
        # Для непрерывного потока нужна более сложная оконная функция с перекрытием.
        # Сейчас: если есть n_fft, берем этот кусок, остаток оставляем в буфере.
        # Сдвигаем буфер на hop_length после обработки.

        while len(self.input_buffer) >= self.n_fft:
            segment_to_process = self.input_buffer[:self.n_fft] # Берем первый полный кадр n_fft

            # Конвертация в мел-спектрограмму
            try:
                mel_spec = wav_to_mel_spectrogram(segment_to_process, self.sample_rate,
                                                 n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels,
                                                 fmin=self.fmin, fmax=self.fmax, power_to_db=self.power_to_db)
                input_mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).to(self.device)
            except Exception as e:
                logger.error(f"Ошибка конвертации в мел-спектрограмму: {e}", exc_info=False)
                self.input_buffer = self.input_buffer[self.hop_length:] # Сдвигаем буфер, чтобы избежать зацикливания на плохих данных
                continue # Пропускаем этот чанк

            # Инференс Генератора
            try:
                with torch.no_grad():
                    # target_speaker_embedding уже подготовлен в __init__
                    converted_mel = self.generator(input_mel_tensor, self.target_speaker_embedding)
            except Exception as e:
                logger.error(f"Ошибка инференса генератора: {e}", exc_info=False)
                self.input_buffer = self.input_buffer[self.hop_length:]
                continue

            # Инференс Вокодера
            try:
                # vocoder.mel_to_wav ожидает (B, C, T) или (C, T), наш converted_mel (1, C, T)
                # squeeze(0) делает (C,T). cpu() и detach() для numpy.
                vocoded_chunk_tensor = self.vocoder.mel_to_wav(converted_mel) # vocoder.py уже делает squeeze(0).cpu()
                processed_chunk_np = vocoded_chunk_tensor.numpy() if isinstance(vocoded_chunk_tensor, torch.Tensor) else vocoded_chunk_tensor

                # Добавляем обработанный чанк в выходную очередь
                self.output_audio_buffer.append(processed_chunk_np)
            except Exception as e:
                logger.error(f"Ошибка инференса вокодера: {e}", exc_info=False)
                # Не добавляем ничего в выходной буфер при ошибке вокодера

            # Сдвигаем входной буфер на hop_length для следующего окна
            # Это базовая форма скользящего окна. Для более качественного результата
            # может понадобиться overlap-add на выходе вокодера.
            self.input_buffer = self.input_buffer[self.hop_length:]

        # 4. Заполнение выходного буфера outdata
        # Собираем данные из self.processed_audio_for_output и self.output_audio_buffer

        # Сначала добавляем новые обработанные чанки в наш 'длинный' буфер
        while len(self.output_audio_buffer) > 0:
            self.processed_audio_for_output = np.concatenate((self.processed_audio_for_output, self.output_audio_buffer.popleft()))

        if len(self.processed_audio_for_output) >= frames:
            output_block = self.processed_audio_for_output[:frames]
            self.processed_audio_for_output = self.processed_audio_for_output[frames:]
            outdata[:] = output_block.reshape(frames, 1) # Моно выход
        else:
            # Недостаточно данных, заполняем тишиной
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)


    def start(self):
        input_dev_idx = self.rt_config.get('input_device_index')
        output_dev_idx = self.rt_config.get('output_device_index')

        logger.info(f"Попытка запуска аудиопотока в реальном времени...")
        logger.info(f"  ID устройства ввода: {input_dev_idx}, ID устройства вывода: {output_dev_idx}")
        logger.info(f"  Частота дискретизации: {self.sample_rate}, Размер блока потока (кадров): {self.stream_block_size}")
        logger.info(f"  Параметры Mel: n_fft={self.n_fft}, hop_length={self.hop_length}")

        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate, blocksize=self.stream_block_size,
                device=(input_dev_idx, output_dev_idx), channels=1, dtype='float32',
                callback=self._audio_callback, latency=self.rt_config.get('latency', 'low') # 'low', 'high', or float seconds
            )
            self.stream.start()
            logger.info("Аудиопоток запущен. Нажмите Ctrl+C для остановки.")
            while self.stream and self.stream.active: time.sleep(0.1) # Держим основной поток живым
        except Exception as e:
            logger.error(f"Ошибка запуска аудиопотока: {e}", exc_info=True)
            if self.stream: self.stream.close()
            self.stream = None

    def stop(self):
        logger.info("Остановка аудиопотока...")
        if self.stream:
            self.stream.stop(); self.stream.close()
            logger.info("Аудиопоток остановлен и закрыт.")
        self.stream = None

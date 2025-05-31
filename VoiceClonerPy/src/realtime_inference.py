import torch
import sounddevice as sd
import numpy as np
import logging
import time
import collections
import os # Для os.path.exists

from .models.stargan_vc import Generator
from .models.vocoder import HiFiGANVocoder
from .audio_utils import wav_to_mel_spectrogram

try:
    import noisereduce
except ImportError:
    noisereduce = None
    # print("Предупреждение: библиотека 'noisereduce' не найдена. Подавление шума будет недоступно.") # Заменено на логгер

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(); formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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

        logger.info("Загрузка модели Генератора для реального времени...")
        self.generator = Generator(config=self.model_config).to(self.device)

        gen_path_key_map = {"male_to_female": "fine_tuned_model_path_female_generator", "female_to_male": "fine_tuned_model_path_male_generator", "specific": "specific_generator_checkpoint_path"}
        direction = self.rt_config.get('conversion_direction', 'specific')
        gen_model_path_key = gen_path_key_map.get(direction, 'specific_generator_checkpoint_path')
        # Используем get с возможностью fallback на общий путь к генератору, если он есть в model_config
        generator_checkpoint_path = self.rt_config.get(gen_model_path_key) or self.model_config.get('generator_checkpoint_path')


        if generator_checkpoint_path and os.path.exists(generator_checkpoint_path):
            checkpoint_g = torch.load(generator_checkpoint_path, map_location=self.device)
            if 'generator_state_dict' in checkpoint_g: self.generator.load_state_dict(checkpoint_g['generator_state_dict'])
            else: self.generator.load_state_dict(checkpoint_g)
            logger.info(f"Генератор загружен из {generator_checkpoint_path}")
        else:
            msg = f"Чекпоинт генератора не найден: {generator_checkpoint_path}. Конвертация в реальном времени не будет работать."
            logger.error(msg); raise FileNotFoundError(msg)
        self.generator.eval()

        logger.info("Загрузка модели Вокодера HiFi-GAN...")
        vocoder_checkpoint = self.model_config['vocoder']['checkpoint_path']
        self.vocoder = HiFiGANVocoder(checkpoint_path=vocoder_checkpoint, config=full_config)
        if self.vocoder.model is None: logger.warning("Модель HiFi-GAN в вокодере не загружена (None). Воспроизведение будет 'dummy' аудио.")

        self.noise_suppressor = noisereduce if self.rt_config.get('noise_suppression_enabled', False) and noisereduce else None
        if self.noise_suppressor: logger.info("Подавление шума включено.")
        else: logger.info("Подавление шума выключено или библиотека 'noisereduce' недоступна.")

        self.sample_rate = self.data_config['sample_rate']
        self.block_size_samples = self.rt_config.get('buffer_size_samples', int(0.1 * self.sample_rate)) # Размер блока в сэмплах

        self.input_buffer = np.array([], dtype=np.float32) # Буфер для накопления входных аудиоданных
        self.output_audio_buffer = collections.deque() # Очередь для обработанных (вокодированных) аудиочанков

        # Определение ID целевого диктора
        if direction == 'male_to_female': self.target_speaker_id = self.rt_config.get('target_female_speaker_id',0)
        elif direction == 'female_to_male': self.target_speaker_id = self.rt_config.get('target_male_speaker_id',1)
        else: self.target_speaker_id = self.rt_config.get('specific_target_speaker_id',0)
        logger.info(f"ID целевого диктора для конвертации: {self.target_speaker_id}")

        # Предварительное вычисление эмбеддинга целевого диктора
        temp_speaker_embedding_layer = nn.Embedding(self.model_config['num_speakers'], self.model_config['speaker_embedding_dim']).to(self.device)
        self.target_speaker_embedding = temp_speaker_embedding_layer(torch.tensor([self.target_speaker_id], dtype=torch.long).to(self.device))
        logger.info(f"Эмбеддинг целевого диктора подготовлен. Форма: {self.target_speaker_embedding.shape}")
        self.stream = None

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        # Колбэк функция для аудиопотока sounddevice
        if status: logger.warning(f"Статус аудиопотока: {status}")
        input_chunk = indata[:, 0] # Предполагаем моно ввод

        if self.noise_suppressor:
            try: input_chunk = self.noise_suppressor.reduce_noise(y=input_chunk, sr=self.sample_rate, quiet=True)
            except Exception as e: logger.error(f"Ошибка при подавлении шума: {e}")

        # TODO: Реализовать корректную потоковую обработку:
        # 1. Добавить input_chunk в self.input_buffer.
        # 2. Если в self.input_buffer достаточно данных для кадра/окна мел-спектрограммы:
        #    a. Извлечь окно из self.input_buffer (с возможным перекрытием).
        #    b. Конвертировать окно в мел-спектрограмму.
        #    c. Выполнить инференс генератора.
        #    d. Вокодировать выход генератора.
        #    e. Добавить вокодированный чанк в self.output_audio_buffer.
        # 3. Если в self.output_audio_buffer достаточно данных для outdata:
        #    a. Извлечь данные и заполнить outdata.
        # 4. Иначе, заполнить outdata нулями (тишина).
        logger.debug(f"Колбэк аудио: форма indata {indata.shape}, кадров {frames}")

        # Пока что простой проброс или тишина (плейсхолдер)
        if len(self.output_audio_buffer) >= frames:
            # Эта логика не совсем корректна для непрерывного потока, т.к. popleft может вернуть чанк другого размера.
            # Нужен более сложный механизм буферизации на выходе.
            processed_chunk_list = [self.output_audio_buffer.popleft() for _ in range(min(frames, len(self.output_audio_buffer)))]
            if processed_chunk_list:
                processed_chunk = np.concatenate(processed_chunk_list)[:frames] #Обрезаем/склеиваем
                outdata[:len(processed_chunk)] = processed_chunk.reshape(-1,1)
                if len(processed_chunk) < frames:
                     outdata[len(processed_chunk):] = np.zeros((frames - len(processed_chunk), outdata.shape[1]), dtype=np.float32)
            else: outdata[:] = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
        else:
            outdata[:] = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
            # Для теста можно добавлять тишину или входные данные в выходной буфер
            # self.output_audio_buffer.append(np.zeros_like(input_chunk))


    def start(self):
        input_dev_idx = self.rt_config.get('input_device_index')
        output_dev_idx = self.rt_config.get('output_device_index')

        logger.info(f"Попытка запуска аудиопотока в реальном времени...")
        logger.info(f"  ID устройства ввода: {input_dev_idx}, ID устройства вывода: {output_dev_idx}")
        logger.info(f"  Частота дискретизации: {self.sample_rate}, Размер блока (кадров): {self.block_size_samples}")

        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate, blocksize=self.block_size_samples,
                device=(input_dev_idx, output_dev_idx), channels=1, dtype='float32',
                callback=self._audio_callback, latency='low'
            )
            self.stream.start()
            logger.info("Аудиопоток запущен. Нажмите Ctrl+C для остановки.")
            while self.stream and self.stream.active: time.sleep(0.1)
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

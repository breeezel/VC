import torch
import torch.optim as optim
import time
import os
import logging
import numpy as np

# Импорт модулей проекта
from .losses import (
    calculate_generator_adv_loss, calculate_discriminator_adv_loss,
    calculate_reconstruction_loss, calculate_identity_mapping_loss,
    calculate_speaker_classification_loss_generator, calculate_speaker_classification_loss_discriminator
)
from ..models.stargan_vc import Generator, Discriminator
from ..data_loader import load_wav # Для валидации
from ..evaluation import calculate_mcd, calculate_f0_rmse # Для валидации

class Trainer:
    def __init__(self, config, generator, discriminator,
                 train_dataloader, val_dataloader,
                 device):

        self.config = config
        self.train_config = config['training']
        self.model_config = config['model']
        self.data_config = config['data']

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_dataloader = train_dataloader
        self.val_loader = val_dataloader
        self.device = device

        self.checkpoint_dir = os.path.join(self.train_config['checkpoint_dir'], config['project']['experiment_name'])
        self.log_file_path = self.train_config['log_file_path']
        self.log_interval = self.train_config['log_interval']

        self.current_epoch = 0 # Будет обновлено при возобновлении
        # Инициализация на основе основной метрики (например, MCD, чем меньше, тем лучше)
        self.primary_metric = config.get('evaluation', {}).get('primary_metric', 'mcd')
        self.best_metric_value = float('inf') if self.primary_metric in ['mcd', 'f0_rmse'] else float('-inf')

        # Оптимизаторы
        g_lr = self.train_config['learning_rate_g']; d_lr = self.train_config['learning_rate_d']
        betas = tuple(self.train_config.get('optimizer_betas', [0.5, 0.999]))
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=g_lr, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=betas)

        # Веса функций потерь
        self.lambda_identity = self.train_config.get('lambda_identity', 1.0)
        self.lambda_reconstruction = self.train_config.get('lambda_reconstruction', 1.0)
        self.lambda_g_adv = self.train_config.get('lambda_g_adv', 1.0)
        self.lambda_g_cls = self.train_config.get('lambda_g_cls', 1.0)
        self.lambda_d_adv = self.train_config.get('lambda_d_adv', 1.0)
        self.lambda_d_cls = self.train_config.get('lambda_d_cls', 1.0)

        # Настройка логгера
        self.logger = logging.getLogger(self.__class__.__name__); self.logger.setLevel(logging.INFO)
        for handler in self.logger.handlers[:]: self.logger.removeHandler(handler); handler.close()
        console_handler = logging.StreamHandler(); console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); self.logger.addHandler(console_handler)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8'); file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); self.logger.addHandler(file_handler)
        self.logger.info("Trainer инициализирован.")
        self.logger.info(f"Гиперпараметры: {self.train_config}") # Выводим только секцию training для краткости
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Директория для чекпоинтов: {self.checkpoint_dir}")

        # Возобновление с чекпоинта
        resume_epoch_num = self.train_config.get('resume_from_epoch', 0)
        if resume_epoch_num > 0:
            self.logger.info(f"Попытка возобновить обучение с эпохи {resume_epoch_num}.")
            loaded_epoch = self.load_checkpoint(epoch_to_load=resume_epoch_num)
            if loaded_epoch > 0: self.current_epoch = loaded_epoch
            else: self.logger.warning(f"Не удалось возобновить с эпохи {resume_epoch_num}. Начинаем с нуля."); self.current_epoch = 0
        else: self.current_epoch = 0

        # Вокодер для валидации (плейсхолдер)
        if self.val_loader:
            from ..models.vocoder import HiFiGANVocoder
            vocoder_checkpoint = self.model_config['vocoder']['checkpoint_path']
            self.vocoder_eval = HiFiGANVocoder(checkpoint_path=vocoder_checkpoint, config=config)
            if self.vocoder_eval.model is None:
                self.logger.warning("Валидация: Модель HiFi-GAN в vocoder_eval не загружена (None). Метрики будут использовать 'dummy' аудио.")

    def _get_speaker_embedding(self, speaker_id):
        # Получение эмбеддинга диктора
        if not hasattr(self, 'speaker_embedding_lookup'):
            num_speakers = self.model_config['num_speakers']
            speaker_embedding_dim = self.model_config['speaker_embedding_dim']
            self.speaker_embedding_lookup = nn.Embedding(num_speakers, speaker_embedding_dim).to(self.device)
            self.logger.info(f"Инициализирован слой эмбеддингов дикторов: кол-во дикторов={num_speakers}, размерность={speaker_embedding_dim}")
        return self.speaker_embedding_lookup(speaker_id)

    def train_epoch(self, epoch_num): # epoch_num нумеруется с 0
        self.generator.train(); self.discriminator.train()
        total_g_loss_epoch, total_d_loss_epoch = 0.0, 0.0
        epoch_start_time = time.time()
        num_epochs_total = self.train_config['num_epochs']

        self.logger.info(f"--- Начало Эпохи {epoch_num + 1}/{num_epochs_total} ---")

        for i, batch in enumerate(self.train_dataloader):
            batch_start_time = time.time()
            # Предполагаемая структура батча: (mel_real, speaker_id_real)
            mel_real, speaker_id_real = batch
            mel_real, speaker_id_real = mel_real.to(self.device), speaker_id_real.to(self.device)

            # Случайные целевые ID дикторов для генерации фейков
            rand_indices = torch.randperm(speaker_id_real.size(0)).to(self.device)
            speaker_id_target_fake = speaker_id_real[rand_indices]

            speaker_emb_real = self._get_speaker_embedding(speaker_id_real)
            speaker_emb_target_fake = self._get_speaker_embedding(speaker_id_target_fake)

            # --- Обучение Дискриминатора ---
            self.optimizer_d.zero_grad()
            real_d_output, real_d_speaker_logits = self.discriminator(mel_real)
            fake_mel = self.generator(mel_real, speaker_emb_target_fake)
            fake_d_output, _ = self.discriminator(fake_mel.detach()) # detach, чтобы не обучать генератор

            d_adv_loss = calculate_discriminator_adv_loss(real_d_output, fake_d_output)
            d_speaker_class_loss = calculate_speaker_classification_loss_discriminator(speaker_id_real, real_d_speaker_logits)
            d_loss = self.lambda_d_adv * d_adv_loss + self.lambda_d_cls * d_speaker_class_loss
            d_loss.backward(); self.optimizer_d.step()

            # --- Обучение Генератора ---
            self.optimizer_g.zero_grad()
            # Повторно используем fake_mel, но теперь D - часть графа G
            fake_d_output_for_g, fake_d_speaker_logits_for_g = self.discriminator(fake_mel)

            g_adv_loss = calculate_generator_adv_loss(fake_d_output_for_g)
            g_speaker_class_loss = calculate_speaker_classification_loss_generator(speaker_id_target_fake, fake_d_speaker_logits_for_g)
            identity_reconstructed_mel = self.generator(mel_real, speaker_emb_real) # G(X_real, Emb_real) -> X_real
            identity_loss = calculate_identity_mapping_loss(mel_real, identity_reconstructed_mel)
            reconstructed_source_mel = self.generator(fake_mel, speaker_emb_real) # G(Fake_mel, Emb_real) -> X_real (цикл)
            reconstruction_loss_src = calculate_reconstruction_loss(mel_real, reconstructed_source_mel)

            g_loss = (self.lambda_g_adv * g_adv_loss + self.lambda_g_cls * g_speaker_class_loss +
                      self.lambda_identity * identity_loss + self.lambda_reconstruction * reconstruction_loss_src)
            g_loss.backward(); self.optimizer_g.step()

            total_g_loss_epoch += g_loss.item(); total_d_loss_epoch += d_loss.item()

            if (i + 1) % self.log_interval == 0:
                batch_duration = time.time() - batch_start_time
                self.logger.info(f"Эпоха [{epoch_num+1}/{num_epochs_total}], Шаг [{i+1}/{len(self.train_dataloader)}], "
                                 f"D_Loss: {d_loss.item():.4f} (Adv: {d_adv_loss.item():.4f}, Cls: {d_speaker_class_loss.item():.4f}), "
                                 f"G_Loss: {g_loss.item():.4f} (Adv: {g_adv_loss.item():.4f}, Cls: {g_speaker_class_loss.item():.4f}, "
                                 f"Id: {identity_loss.item():.4f}, Rec: {reconstruction_loss_src.item():.4f}), "
                                 f"Время батча: {batch_duration:.2f}с")

        epoch_duration = time.time() - epoch_start_time
        avg_g_loss = total_g_loss_epoch / len(self.train_dataloader) if self.train_dataloader and len(self.train_dataloader) > 0 else 0
        avg_d_loss = total_d_loss_epoch / len(self.train_dataloader) if self.train_dataloader and len(self.train_dataloader) > 0 else 0
        self.logger.info(f"--- Эпоха {epoch_num+1} Завершена --- Сред. G_Loss: {avg_g_loss:.4f}, Сред. D_Loss: {avg_d_loss:.4f}, Время: {epoch_duration:.2f}с")

        if (epoch_num + 1) % self.train_config.get('save_epoch_interval', 1) == 0:
             self.save_checkpoint(epoch=epoch_num, metrics={'train_g_loss': avg_g_loss})

        return {'avg_g_loss': avg_g_loss, 'avg_d_loss': avg_d_loss}

    def evaluate_epoch(self, epoch_num):
        if not self.val_loader:
            self.logger.warning("Валидационный DataLoader не предоставлен. Пропуск оценки.")
            return None
        if not hasattr(self, 'vocoder_eval') or self.vocoder_eval is None:
             self.logger.warning("Вокодер для оценки не инициализирован. Пропуск оценки.")
             return None

        self.logger.info(f"--- Начало Валидации для Эпохи {epoch_num + 1} ---")
        self.generator.eval()

        total_mcd, total_f0_rmse, count = 0.0, 0.0, 0
        eval_config = self.config.get('evaluation', {})
        mfcc_params = eval_config.get('mfcc_params') # Будут None, если не заданы
        f0_params = eval_config.get('f0_params')

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_loader):
                # Валидационный даталоадер должен возвращать: (source_mel, source_speaker_id, target_speaker_id, target_wav_path)
                source_mel_batch = batch_data[0].to(self.device)
                target_speaker_id_batch = batch_data[2].to(self.device) # ID для конверсии
                target_wav_path_batch = batch_data[3] # Список путей

                target_speaker_emb_batch = self._get_speaker_embedding(target_speaker_id_batch)
                converted_mel_batch = self.generator(source_mel_batch, target_speaker_emb_batch)

                for j in range(converted_mel_batch.size(0)):
                    converted_mel = converted_mel_batch[j]
                    target_wav_path = target_wav_path_batch[j]

                    converted_audio_tensor = self.vocoder_eval.mel_to_wav(converted_mel.unsqueeze(0))
                    converted_audio_data = converted_audio_tensor.squeeze().cpu().numpy()

                    target_audio_data, sr_target = load_wav(target_wav_path, sample_rate=self.data_config['sample_rate'])
                    if target_audio_data is None:
                        self.logger.warning(f"Не удалось загрузить целевое аудио {target_wav_path} для валидации. Пропуск.")
                        continue

                    if converted_audio_data.ndim > 1: converted_audio_data = converted_audio_data.flatten()
                    if target_audio_data.ndim > 1: target_audio_data = target_audio_data.flatten()

                    mcd = calculate_mcd(converted_audio_data, target_audio_data, self.data_config['sample_rate'], mfcc_params)
                    f0_rmse = calculate_f0_rmse(converted_audio_data, target_audio_data, self.data_config['sample_rate'], f0_params)

                    total_mcd += mcd; total_f0_rmse += f0_rmse; count += 1
                    if i % self.log_interval == 0 and j == 0 :
                        self.logger.info(f"  Валидация [{i*self.val_loader.batch_size+j+1}/{len(self.val_loader.dataset)}]: MCD={mcd:.4f}, F0-RMSE={f0_rmse:.2f}")

        avg_mcd = total_mcd / count if count > 0 else float('inf')
        avg_f0_rmse = total_f0_rmse / count if count > 0 else float('inf')
        self.logger.info(f"--- Эпоха {epoch_num + 1} Результаты Валидации ---")
        self.logger.info(f"Средний MCD: {avg_mcd:.4f}, Средний F0-RMSE: {avg_f0_rmse:.2f} (по {count} сэмплам)")
        return {'mcd': avg_mcd, 'f0_rmse': avg_f0_rmse}

    def run_training(self):
        num_epochs = self.train_config['num_epochs']
        start_epoch_for_loop = self.current_epoch
        self.logger.info(f"Запуск цикла обучения с эпохи {start_epoch_for_loop + 1} до {num_epochs}.")

        for epoch in range(start_epoch_for_loop, num_epochs):
            self.train_epoch(epoch)
            val_metrics = None
            if self.val_loader: val_metrics = self.evaluate_epoch(epoch)

            if val_metrics:
                metric_to_check = val_metrics.get(self.primary_metric)
                if metric_to_check is not None:
                    is_better = (metric_to_check < self.best_metric_value) if self.primary_metric in ['mcd', 'f0_rmse'] else (metric_to_check > self.best_metric_value)
                    if is_better:
                        self.best_metric_value = metric_to_check
                        best_model_prefix = os.path.join(self.checkpoint_dir, f"model_best_{self.primary_metric}")
                        self.save_checkpoint(epoch=epoch, full_file_path_prefix=best_model_prefix, metrics=val_metrics)
                        self.logger.info(f"Новая лучшая модель сохранена на основе {self.primary_metric}: {self.best_metric_value:.4f}")
                else: self.logger.warning(f"Основная метрика '{self.primary_metric}' не найдена в результатах валидации. Не могу сохранить лучшую модель.")
        self.logger.info("Обучение завершено.")

    def save_checkpoint(self, epoch, full_file_path_prefix=None, metrics=None, save_generator_only=False):
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if full_file_path_prefix:
            g_path, d_path = f"{full_file_path_prefix}_generator.pth", f"{full_file_path_prefix}_discriminator.pth"
            opt_g_path, opt_d_path = f"{full_file_path_prefix}_optimizer_g.pth", f"{full_file_path_prefix}_optimizer_d.pth"
            log_name = os.path.basename(full_file_path_prefix)
        else:
            suffix = f"epoch_{epoch+1}"
            if metrics: suffix += "".join([f"_{k}_{v:.4f}" for k,v in metrics.items() if isinstance(v,(int,float))])
            g_path, d_path = os.path.join(self.checkpoint_dir, f'generator_{suffix}.pth'), os.path.join(self.checkpoint_dir, f'discriminator_{suffix}.pth')
            opt_g_path, opt_d_path = os.path.join(self.checkpoint_dir, f'optimizer_g_{suffix}.pth'), os.path.join(self.checkpoint_dir, f'optimizer_d_{suffix}.pth')
            log_name = f"эпоха {epoch+1}"
        try:
            torch.save(self.generator.state_dict(), g_path)
            if not save_generator_only:
                torch.save(self.discriminator.state_dict(), d_path); torch.save(self.optimizer_g.state_dict(), opt_g_path); torch.save(self.optimizer_d.state_dict(), opt_d_path)
            self.logger.info(f"Чекпоинт '{log_name}' сохранен. Метрики: {metrics}. Только генератор: {save_generator_only}")
        except Exception as e: self.logger.error(f"Ошибка сохранения чекпоинта '{log_name}': {e}", exc_info=True)

    def load_checkpoint(self, epoch_to_load): # 1-индексированный
        paths_to_check = [os.path.join(self.checkpoint_dir, f'{name}_epoch_{epoch_to_load}.pth') for name in ['generator', 'discriminator', 'optimizer_g', 'optimizer_d']]
        if any(not os.path.exists(p) for p in paths_to_check):
            self.logger.warning(f"Чекпоинт для эпохи {epoch_to_load} не найден полностью в {self.checkpoint_dir}."); return 0
        try:
            self.generator.load_state_dict(torch.load(paths_to_check[0], map_location=self.device))
            self.discriminator.load_state_dict(torch.load(paths_to_check[1], map_location=self.device))
            self.optimizer_g.load_state_dict(torch.load(paths_to_check[2], map_location=self.device))
            self.optimizer_d.load_state_dict(torch.load(paths_to_check[3], map_location=self.device))
            self.logger.info(f"Успешно загружен чекпоинт для эпохи {epoch_to_load} из {self.checkpoint_dir}"); return epoch_to_load
        except Exception as e: self.logger.error(f"Ошибка загрузки чекпоинта эпохи {epoch_to_load}: {e}", exc_info=True); return 0

    def fine_tune_epoch(self, epoch_num, fine_tune_dataloader): # Плейсхолдер
        self.logger.info(f"--- Начало Эпохи Fine-tuning {epoch_num + 1}/{self.config['fine_tuning']['num_epochs']} ---")
        self.logger.warning("Логика fine_tune_epoch - плейсхолдер.")
        self.logger.info(f"--- Эпоха Fine-tuning {epoch_num + 1} Завершена (Плейсхолдер) ---")

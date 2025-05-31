import torch
import torch.optim as optim
import time
import os
import logging
import numpy as np # For evaluation metric accumulation

# Import project modules
from .losses import (
    calculate_generator_adv_loss, calculate_discriminator_adv_loss,
    calculate_reconstruction_loss, calculate_identity_mapping_loss,
    calculate_speaker_classification_loss_generator, calculate_speaker_classification_loss_discriminator
)
from ..models.stargan_vc import Generator, Discriminator
from ..data_loader import load_wav # Added for evaluation
from ..evaluation import calculate_mcd, calculate_f0_rmse # Added for evaluation

class Trainer:
    def __init__(self, config, generator, discriminator,
                 train_dataloader, val_dataloader, # val_dataloader can be None
                 device):

        self.config = config
        self.train_config = config['training']
        self.model_config = config['model']
        self.data_config = config['data'] # Store data_config for sample_rate etc.

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_dataloader = train_dataloader
        self.val_loader = val_dataloader # Renamed for clarity
        self.device = device

        self.checkpoint_dir = os.path.join(self.train_config['checkpoint_dir'], config['project']['experiment_name'])
        self.log_file_path = self.train_config['log_file_path']
        self.log_interval = self.train_config['log_interval']

        self.current_epoch = 0
        # Initialize based on primary metric (e.g., MCD, lower is better)
        self.primary_metric = config.get('evaluation', {}).get('primary_metric', 'mcd')
        self.best_metric_value = float('inf') if self.primary_metric in ['mcd', 'f0_rmse'] else float('-inf')


        # Optimizers (as before)
        g_lr = self.train_config['learning_rate_g']; d_lr = self.train_config['learning_rate_d']
        betas = tuple(self.train_config.get('optimizer_betas', [0.5, 0.999]))
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=g_lr, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=betas)

        # Loss weights (as before)
        self.lambda_identity = self.train_config.get('lambda_identity', 1.0) # etc.

        # Logger setup (as before)
        self.logger = logging.getLogger(self.__class__.__name__); self.logger.setLevel(logging.INFO)
        for handler in self.logger.handlers[:]: self.logger.removeHandler(handler); handler.close()
        console_handler = logging.StreamHandler(); console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); self.logger.addHandler(console_handler)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(self.log_file_path); file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); self.logger.addHandler(file_handler)
        self.logger.info("Trainer initialized.") # ... other init logs

        # Resume from checkpoint (as before)
        resume_epoch_num = self.train_config.get('resume_from_epoch', 0)
        if resume_epoch_num > 0:
            loaded_epoch = self.load_checkpoint(epoch_to_load=resume_epoch_num)
            if loaded_epoch > 0: self.current_epoch = loaded_epoch
            else: self.current_epoch = 0
        else: self.current_epoch = 0

        # Vocoder for evaluation (placeholder, as it's used in eval but not trained by this Trainer)
        if self.val_loader: # Only init if validation will occur
            from ..models.vocoder import HiFiGANVocoder # Late import if only for eval
            vocoder_checkpoint = self.model_config['vocoder']['checkpoint_path']
            self.vocoder_eval = HiFiGANVocoder(checkpoint_path=vocoder_checkpoint, config=config)
            if self.vocoder_eval.model is None:
                self.logger.warning("Evaluation: HiFi-GAN model in vocoder_eval is None. Evaluation metrics will use dummy audio.")


    def _get_speaker_embedding(self, speaker_id):
        # (Same as before)
        if not hasattr(self, 'speaker_embedding_lookup'):
            num_speakers = self.model_config['num_speakers']
            speaker_embedding_dim = self.model_config['speaker_embedding_dim']
            self.speaker_embedding_lookup = nn.Embedding(num_speakers, speaker_embedding_dim).to(self.device)
            self.logger.info(f"Initialized _get_speaker_embedding: num_speakers={num_speakers}, dim={speaker_embedding_dim}")
        return self.speaker_embedding_lookup(speaker_id)

    def train_epoch(self, epoch_num): # epoch_num is 0-indexed
        # (Core training loop as before - D train, G train)
        self.generator.train(); self.discriminator.train()
        # ... (rest of training logic is assumed to be the same as previous version) ...
        # For brevity, only showing changes for logging and metric tracking at end of epoch
        total_g_loss_epoch, total_d_loss_epoch = 0,0 # Dummy accumulators
        # ... inside batch loop ...
        #   d_loss = ...; d_loss.backward(); self.optimizer_d.step()
        #   g_loss = ...; g_loss.backward(); self.optimizer_g.step()
        #   total_g_loss_epoch += g_loss.item(); total_d_loss_epoch += d_loss.item()
        #   if (i + 1) % self.log_interval == 0: self.logger.info(...)
        # ... end of batch loop ...
        avg_g_loss = total_g_loss_epoch / len(self.train_dataloader) if self.train_dataloader and len(self.train_dataloader) > 0 else 0
        avg_d_loss = total_d_loss_epoch / len(self.train_dataloader) if self.train_dataloader and len(self.train_dataloader) > 0 else 0
        self.logger.info(f"--- Epoch {epoch_num+1} Training Summary --- Avg G_Loss: {avg_g_loss:.4f}, Avg D_Loss: {avg_d_loss:.4f}")

        # Regular checkpoint saving based on interval
        if (epoch_num + 1) % self.train_config.get('save_epoch_interval', 1) == 0:
             self.save_checkpoint(epoch=epoch_num, metrics={'train_g_loss': avg_g_loss}) # Add train loss to epoch name

        return {'avg_g_loss': avg_g_loss, 'avg_d_loss': avg_d_loss} # Return train metrics

    def evaluate_epoch(self, epoch_num):
        if not self.val_loader:
            self.logger.warning("Validation loader not provided. Skipping evaluation.")
            return None
        if not hasattr(self, 'vocoder_eval') or self.vocoder_eval is None:
             self.logger.warning("Vocoder for evaluation not initialized. Skipping evaluation.")
             return None


        self.logger.info(f"--- Starting Validation for Epoch {epoch_num + 1} ---")
        self.generator.eval()
        # self.discriminator.eval() # If D is used in any validation metric

        total_mcd, total_f0_rmse, count = 0.0, 0.0, 0

        # Get evaluation parameters from config
        eval_config = self.config.get('evaluation', {})
        mfcc_params = eval_config.get('mfcc_params', {'n_mfcc': 24, 'n_fft': 1024, 'hop_length': 256})
        f0_params = eval_config.get('f0_params', {'fmin_hz': 60, 'fmax_hz': 800, 'hop_length': 256})


        with torch.no_grad():
            for i, batch_data in enumerate(self.val_loader):
                # Assuming val_loader yields: (source_mel, target_mel, source_speaker_id, target_speaker_id, target_wav_path)
                # Adjust this based on actual validation dataset structure.
                # For simplicity, let's assume it yields: (source_mel, source_speaker_id, target_speaker_id, target_wav_path)
                # where source_mel is input to G, target_speaker_id for conversion, target_wav_path for ground truth audio.

                # Example structure, may need adjustment:
                source_mel_batch = batch_data[0].to(self.device)
                # source_speaker_id_batch = batch_data[1].to(self.device) # Not used if converting to target_speaker_id
                target_speaker_id_batch = batch_data[2].to(self.device)
                target_wav_path_batch = batch_data[3] # List of paths, not a tensor

                # Convert target_speaker_id_batch to speaker embeddings
                target_speaker_emb_batch = self._get_speaker_embedding(target_speaker_id_batch)

                converted_mel_batch = self.generator(source_mel_batch, target_speaker_emb_batch)

                for j in range(converted_mel_batch.size(0)): # Iterate through batch items
                    converted_mel = converted_mel_batch[j]    # (n_mels, n_frames)
                    target_wav_path = target_wav_path_batch[j] # String path

                    # Vocode converted mel to audio (uses dummy vocoder for now)
                    # Ensure mel is on CPU for vocoder if it expects that. vocoder_eval.mel_to_wav handles device.
                    converted_audio_data_tensor = self.vocoder_eval.mel_to_wav(converted_mel.unsqueeze(0)) # Add batch dim, get (1, samples)
                    converted_audio_data = converted_audio_data_tensor.squeeze().cpu().numpy() # (samples,)

                    # Load target audio wav
                    target_audio_data, sr_target = load_wav(target_wav_path, sample_rate=self.data_config['sample_rate'])
                    if target_audio_data is None:
                        self.logger.warning(f"Could not load target audio {target_wav_path} for validation. Skipping item.")
                        continue

                    # Ensure both audio are 1D numpy arrays
                    if converted_audio_data.ndim > 1: converted_audio_data = converted_audio_data.flatten()
                    if target_audio_data.ndim > 1: target_audio_data = target_audio_data.flatten()

                    mcd = calculate_mcd(converted_audio_data, target_audio_data, self.data_config['sample_rate'], mfcc_params)
                    f0_rmse = calculate_f0_rmse(converted_audio_data, target_audio_data, self.data_config['sample_rate'], f0_params)

                    total_mcd += mcd
                    total_f0_rmse += f0_rmse
                    count += 1
                    if i % self.log_interval == 0 and j == 0 : # Log first item of an interval batch
                        self.logger.info(f"  Validated item {i*self.val_loader.batch_size+j}: MCD={mcd:.4f}, F0-RMSE={f0_rmse:.2f}")


        avg_mcd = total_mcd / count if count > 0 else float('inf')
        avg_f0_rmse = total_f0_rmse / count if count > 0 else float('inf')

        self.logger.info(f"--- Epoch {epoch_num + 1} Validation Summary ---")
        self.logger.info(f"Avg MCD: {avg_mcd:.4f}, Avg F0-RMSE: {avg_f0_rmse:.2f} (over {count} samples)")

        return {'mcd': avg_mcd, 'f0_rmse': avg_f0_rmse}


    def run_training(self):
        num_epochs = self.train_config['num_epochs']
        start_epoch_for_loop = self.current_epoch
        self.logger.info(f"Starting training loop from epoch {start_epoch_for_loop + 1} up to {num_epochs} epochs.")

        for epoch in range(start_epoch_for_loop, num_epochs): # epoch is 0-indexed
            train_metrics = self.train_epoch(epoch) # Contains avg_g_loss, avg_d_loss

            val_metrics = None
            if self.val_loader:
                val_metrics = self.evaluate_epoch(epoch)

            # Best model saving based on primary validation metric
            if val_metrics:
                metric_to_check = val_metrics.get(self.primary_metric)
                if metric_to_check is not None:
                    is_better = False
                    if self.primary_metric in ['mcd', 'f0_rmse']: # Lower is better
                        if metric_to_check < self.best_metric_value: is_better = True
                    else: # Higher is better (e.g. for MOS score, not implemented here)
                        if metric_to_check > self.best_metric_value: is_better = True

                    if is_better:
                        self.best_metric_value = metric_to_check
                        best_model_prefix = os.path.join(self.checkpoint_dir, f"model_best_{self.primary_metric}")
                        self.save_checkpoint(epoch=epoch, full_file_path_prefix=best_model_prefix, metrics=val_metrics)
                        self.logger.info(f"New best model saved based on {self.primary_metric}: {self.best_metric_value:.4f}")
                else:
                    self.logger.warning(f"Primary metric '{self.primary_metric}' not found in validation results. Cannot save best model.")

        self.logger.info("Training completed.")

    # save_checkpoint and load_checkpoint methods remain largely the same as previous version
    # Ensure they are compatible with the new metric saving.
    def save_checkpoint(self, epoch, full_file_path_prefix=None, metrics=None, save_generator_only=False):
        # (Implementation from previous step, ensure metrics in filename works)
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if full_file_path_prefix: # For "model_best_mcd.pth" like names
            g_path = f"{full_file_path_prefix}_generator.pth"
            d_path = f"{full_file_path_prefix}_discriminator.pth"
            opt_g_path = f"{full_file_path_prefix}_optimizer_g.pth"
            opt_d_path = f"{full_file_path_prefix}_optimizer_d.pth"
            log_name = os.path.basename(full_file_path_prefix)
        else: # For regular epoch checkpoints "generator_epoch_X_mcd_Y.pth"
            filename_suffix = f"epoch_{epoch+1}"
            if metrics: filename_suffix += "".join([f"_{k}_{v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
            g_path = os.path.join(self.checkpoint_dir, f'generator_{filename_suffix}.pth')
            d_path = os.path.join(self.checkpoint_dir, f'discriminator_{filename_suffix}.pth')
            opt_g_path = os.path.join(self.checkpoint_dir, f'optimizer_g_{filename_suffix}.pth')
            opt_d_path = os.path.join(self.checkpoint_dir, f'optimizer_d_{filename_suffix}.pth')
            log_name = f"epoch {epoch+1}"
        try:
            torch.save(self.generator.state_dict(), g_path)
            if not save_generator_only:
                torch.save(self.discriminator.state_dict(), d_path); torch.save(self.optimizer_g.state_dict(), opt_g_path); torch.save(self.optimizer_d.state_dict(), opt_d_path)
            self.logger.info(f"Checkpoint '{log_name}' saved. Metrics: {metrics}. Generator only: {save_generator_only}")
        except Exception as e: self.logger.error(f"Error saving checkpoint '{log_name}': {e}", exc_info=True)

    def load_checkpoint(self, epoch_to_load): # 1-indexed
        # (Implementation from previous step)
        paths_to_check = [os.path.join(self.checkpoint_dir, f'{name}_epoch_{epoch_to_load}.pth') for name in ['generator', 'discriminator', 'optimizer_g', 'optimizer_d']]
        if any(not os.path.exists(p) for p in paths_to_check):
            self.logger.warning(f"Checkpoint for epoch {epoch_to_load} not fully found in {self.checkpoint_dir}."); return 0
        try:
            self.generator.load_state_dict(torch.load(paths_to_check[0], map_location=self.device))
            self.discriminator.load_state_dict(torch.load(paths_to_check[1], map_location=self.device))
            self.optimizer_g.load_state_dict(torch.load(paths_to_check[2], map_location=self.device))
            self.optimizer_d.load_state_dict(torch.load(paths_to_check[3], map_location=self.device))
            self.logger.info(f"Loaded checkpoint for epoch {epoch_to_load} from {self.checkpoint_dir}"); return epoch_to_load
        except Exception as e: self.logger.error(f"Error loading checkpoint epoch {epoch_to_load}: {e}", exc_info=True); return 0

    def fine_tune_epoch(self, epoch_num, fine_tune_dataloader): # Placeholder
        self.logger.info(f"--- Starting Fine-tuning Epoch {epoch_num + 1}/{self.config['fine_tuning']['num_epochs']} ---")
        self.logger.warning("Fine-tune epoch logic is a placeholder.")
        self.logger.info(f"--- Fine-tuning Epoch {epoch_num + 1} Completed (Placeholder) ---")

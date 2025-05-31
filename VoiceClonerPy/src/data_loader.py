import os
import librosa
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .audio_utils import wav_to_mel_spectrogram

def load_wav(file_path, sample_rate=None):
    # (Same as before)
    if not os.path.exists(file_path): print(f"Error: File not found at {file_path}"); return None, None
    try: audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=True); return audio_data, sr
    except Exception as e: print(f"Error loading WAV file {file_path}: {e}"); return None, None

def save_wav(file_path, audio_data, sample_rate):
    # (Same as before)
    try: sf.write(file_path, audio_data, sample_rate); return True
    except Exception as e: print(f"Error saving WAV file {file_path}: {e}"); return False

def prepare_fine_tune_data(user_data_dir): # (Same as before)
    if not os.path.isdir(user_data_dir): print(f"Error: Directory not found: {user_data_dir}"); return []
    try: wav_files = [os.path.join(user_data_dir, f) for f in os.listdir(user_data_dir) if f.lower().endswith('.wav')]; return wav_files
    except Exception as e: print(f"Error accessing {user_data_dir}: {e}"); return []

def prepare_base_model_data(corpus_dir): # (Same as before)
    if not os.path.isdir(corpus_dir): print(f"Error: Directory not found: {corpus_dir}"); return []
    try: items = [os.path.join(corpus_dir,i) for i in os.listdir(corpus_dir) if i.lower().endswith('.wav')]; return items # Simplified to wav files
    except Exception as e: print(f"Error accessing {corpus_dir}: {e}"); return []


class VoiceDataset(Dataset):
    def __init__(self, data_paths_and_ids, data_config, is_validation=False): # data_paths_and_ids is now list of tuples
        # For training, data_paths_and_ids = [(wav_path1, speaker_id1), (wav_path2, speaker_id2), ...]
        # For validation, it could be [(src_wav_path, trg_wav_path, src_spk_id, trg_spk_id), ...]
        # Or, more simply for this task: [(source_audio_for_mel_path, target_reference_audio_path, source_speaker_id, target_speaker_id_for_conversion)]
        # The key is that for validation, we need target_reference_audio_path for evaluation metrics.
        self.data_entries = data_paths_and_ids # List of tuples
        self.data_config = data_config
        self.is_validation = is_validation # Flag to change __getitem__ behavior

        self.sample_rate = data_config['sample_rate'] # etc. (as before)
        self.n_fft = data_config['n_fft']; self.hop_length = data_config['hop_length']; self.n_mels = data_config['n_mels']
        self.fmin = data_config.get('fmin',0); self.fmax = data_config.get('fmax', self.sample_rate/2); self.power_to_db = data_config.get('power_to_db',True)

        print(f"VoiceDataset initialized with {len(self.data_entries)} samples. Validation mode: {self.is_validation}")

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]

        if self.is_validation:
            # Example validation entry: (source_mel_audio_path, target_eval_audio_path, source_speaker_id, target_speaker_id_for_conversion)
            # This structure needs to be prepared by the script that calls this dataset.
            source_audio_path, target_eval_audio_path, source_speaker_id, target_speaker_id_for_conversion = entry

            # Process source audio for mel spectrogram (input to generator)
            source_audio_data, _ = load_wav(source_audio_path, sample_rate=self.sample_rate)
            if source_audio_data is None: # Handle loading error
                return torch.zeros((self.n_mels, 128)), torch.tensor(source_speaker_id), torch.tensor(target_speaker_id_for_conversion), "dummy_path_error.wav"

            source_mel_np = wav_to_mel_spectrogram(source_audio_data, self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                                                   n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power_to_db=self.power_to_db)
            source_mel_tensor = torch.from_numpy(source_mel_np).float()

            # Return source_mel, source_speaker_id (for potential future use), target_speaker_id (for G), and target_eval_audio_path
            return source_mel_tensor, torch.tensor(source_speaker_id, dtype=torch.long), \
                   torch.tensor(target_speaker_id_for_conversion, dtype=torch.long), target_eval_audio_path
        else: # Training mode
            # Entry: (audio_path, speaker_id)
            audio_path, speaker_id = entry
            audio_data, _ = load_wav(audio_path, sample_rate=self.sample_rate)
            if audio_data is None: # Handle loading error
                return torch.zeros((self.n_mels, 128)), torch.tensor(speaker_id, dtype=torch.long) # Return dummy data for this item

            mel_spectrogram_np = wav_to_mel_spectrogram(audio_data, self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                                                        n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power_to_db=self.power_to_db)
            mel_tensor = torch.from_numpy(mel_spectrogram_np).float()
            return mel_tensor, torch.tensor(speaker_id, dtype=torch.long)

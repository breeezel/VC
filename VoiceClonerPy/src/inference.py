import torch
import logging
import os
import numpy as np

from .models.stargan_vc import Generator
from .models.vocoder import HiFiGANVocoder
from .data_loader import load_wav, save_wav # Assuming load_wav and save_wav are in data_loader.py
from .audio_utils import wav_to_mel_spectrogram

# Setup basic logger for inference
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers: # Add handler only if no handlers are configured
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def convert_voice_from_file(full_config, generator_model_path, input_wav_path, output_wav_path, target_speaker_id_for_generator):
    """
    Converts voice from an input WAV file to a target speaker's voice.
    """
    logger.info(f"Starting voice conversion for: {input_wav_path}")
    logger.info(f"Generator model path: {generator_model_path}")
    logger.info(f"Target speaker ID for generator: {target_speaker_id_for_generator}")

    try:
        # Determine device
        if full_config['training']['device'] == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        logger.info(f"Using device: {device}")

        # Instantiate Generator
        # Generator now takes the 'model' part of the config
        generator = Generator(config=full_config['model']).to(device)

        # Load generator state dictionary
        logger.info(f"Loading generator state from: {generator_model_path}")
        if not os.path.exists(generator_model_path):
            logger.error(f"Generator checkpoint not found: {generator_model_path}")
            return False

        # Checkpoint might be saved directly or within a dict (e.g., {'generator_state_dict': ...})
        # Standardizing to load directly if it's a state_dict, or look for common keys.
        checkpoint = torch.load(generator_model_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        elif 'model_state_dict' in checkpoint and 'generator' in checkpoint['model_state_dict']: # Old format
             generator.load_state_dict(checkpoint['model_state_dict']['generator'])
        elif 'state_dict' in checkpoint: # Another common pattern
            generator.load_state_dict(checkpoint['state_dict'])
        else: # Assume the checkpoint is the state_dict itself
            generator.load_state_dict(checkpoint)

        generator.eval()
        logger.info("Generator model loaded and set to eval mode.")

        # Instantiate Vocoder
        # Vocoder __init__ now takes checkpoint_path and the full_config
        vocoder_checkpoint_path = full_config['model']['vocoder']['checkpoint_path']
        if not vocoder_checkpoint_path or not os.path.exists(vocoder_checkpoint_path):
            logger.warning(f"Vocoder checkpoint path not found or not specified: {vocoder_checkpoint_path}. Vocoder might return dummy audio.")
        vocoder = HiFiGANVocoder(checkpoint_path=vocoder_checkpoint_path, config=full_config)
        # Vocoder's internal model is already on its device if loaded

        # Load source audio
        logger.info(f"Loading source audio: {input_wav_path}")
        audio_data, sr = load_wav(input_wav_path, sample_rate=full_config['data']['sample_rate'])
        if audio_data is None:
            logger.error(f"Failed to load audio from {input_wav_path}")
            return False
        logger.info(f"Audio loaded. Sample rate: {sr}, Duration: {len(audio_data)/sr:.2f}s")

        # Convert audio to mel-spectrogram
        logger.info("Converting audio to mel-spectrogram...")
        mel_spec = wav_to_mel_spectrogram(
            audio_data,
            sample_rate=sr, # Use actual sample rate from loaded audio (should match config)
            n_fft=full_config['data']['n_fft'],
            hop_length=full_config['data']['hop_length'],
            n_mels=full_config['data']['n_mels'],
            fmin=full_config['data']['fmin'],
            fmax=full_config['data']['fmax'],
            power_to_db=full_config['data']['power_to_db']
        )
        logger.info(f"Mel-spectrogram created. Shape: {mel_spec.shape}")

        # Prepare mel tensor for generator
        input_mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).to(device) # (1, n_mels, num_frames)

        # Prepare target speaker embedding tensor
        # This assumes Generator's SpeakerAdaptationMLP has an nn.Embedding layer for integer IDs.
        # Or, that _get_speaker_embedding is used internally if speaker_id is passed.
        # The Generator model from previous step expects speaker_embedding_target directly.
        # So, we need an embedding layer here, or pass ID if generator handles it.
        # For consistency with Trainer, let's create an embedding on the fly.
        # This is a simplification for inference; a shared embedding layer or lookup is better.
        num_speakers = full_config['model']['num_speakers']
        speaker_embedding_dim = full_config['model']['speaker_embedding_dim']
        temp_speaker_embedding_layer = nn.Embedding(num_speakers, speaker_embedding_dim).to(device)
        target_speaker_emb_tensor = temp_speaker_embedding_layer(torch.tensor([target_speaker_id_for_generator], dtype=torch.long).to(device))

        logger.info(f"Target speaker embedding tensor created. Shape: {target_speaker_emb_tensor.shape}")

        # Perform inference
        logger.info("Performing voice conversion (Generator inference)...")
        with torch.no_grad():
            output_mel_tensor = generator(input_mel_tensor, target_speaker_emb_tensor)
        logger.info(f"Output mel-spectrogram from generator. Shape: {output_mel_tensor.shape}")

        # Convert output mel-spectrogram to waveform using Vocoder
        logger.info("Converting output mel-spectrogram to waveform (Vocoder inference)...")
        # Vocoder expects mel on its device, output_mel_tensor is already on self.device
        # Ensure output_mel_tensor is (B, C, T) or (C, T) for vocoder
        # If generator outputs (B,C,T) and B=1, squeeze it.
        # Vocoder.mel_to_wav should handle (B,C,T) or (C,T)
        if output_mel_tensor.ndim == 3 and output_mel_tensor.size(0) == 1:
             processed_output_mel = output_mel_tensor.squeeze(0) # (n_mels, num_frames)
        else:
            processed_output_mel = output_mel_tensor # If already (n_mels, num_frames) or batched

        output_waveform = vocoder.mel_to_wav(processed_output_mel.cpu()) # Vocoder might expect CPU tensor
        logger.info(f"Output waveform generated. Shape: {output_waveform.shape}, Type: {output_waveform.dtype}")

        # Save output waveform
        logger.info(f"Saving output waveform to: {output_wav_path}")
        # Ensure output_waveform is a NumPy array for save_wav
        if isinstance(output_waveform, torch.Tensor):
            output_waveform_np = output_waveform.detach().cpu().numpy()
        else:
            output_waveform_np = output_waveform # If already numpy

        # Ensure output dir exists
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

        success = save_wav(output_wav_path, output_waveform_np, full_config['data']['sample_rate'])
        if success:
            logger.info("Voice conversion completed successfully.")
        else:
            logger.error("Failed to save output waveform.")
        return success

    except Exception as e:
        logger.error(f"An error occurred during voice conversion: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    # Example usage (requires a config file and models)
    print("This script is intended to be called from run.py or programmatically.")
    # Example:
    # config = load_config("path_to_your_config.yaml")
    # if config:
    #     convert_voice_from_file(
    #         full_config=config,
    #         generator_model_path="path_to_generator.pth",
    #         input_wav_path="path_to_source.wav",
    #         output_wav_path="output/converted_audio.wav",
    #         target_speaker_id_for_generator=0 # Example target speaker ID
    #     )

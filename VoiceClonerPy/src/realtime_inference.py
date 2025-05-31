import torch
import sounddevice as sd
import numpy as np
import logging
import time
import collections # For output buffer

from .models.stargan_vc import Generator
from .models.vocoder import HiFiGANVocoder # Assuming HiFiGANVocoder is updated for config
from .audio_utils import wav_to_mel_spectrogram

# Optional noise reduction
try:
    import noisereduce
except ImportError:
    noisereduce = None
    print("Warning: 'noisereduce' library not found. Noise suppression will be unavailable.")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class RealTimeVoiceConverter:
    def __init__(self, full_config):
        self.config = full_config
        self.rt_config = full_config['inference_realtime']
        self.data_config = full_config['data']
        self.model_config = full_config['model']

        self.device = torch.device(full_config['training']['device'] if torch.cuda.is_available() and full_config['training']['device'] == 'cuda' else 'cpu')
        logger.info(f"RealTimeVoiceConverter using device: {self.device}")

        # Load Generator
        logger.info("Loading Generator model for real-time inference...")
        self.generator = Generator(config=self.model_config).to(self.device) # Generator expects model part of config

        # Determine generator path based on conversion direction
        gen_path_key_map = {
            "male_to_female": "fine_tuned_model_path_female_generator",
            "female_to_male": "fine_tuned_model_path_male_generator",
            "specific": "specific_generator_checkpoint_path"
        }
        direction = self.rt_config.get('conversion_direction', 'specific')
        gen_model_path_key = gen_path_key_map.get(direction, 'specific_generator_checkpoint_path')
        generator_checkpoint_path = self.rt_config.get(gen_model_path_key, self.model_config.get('generator_checkpoint_path')) # Fallback to a general G path

        if generator_checkpoint_path and os.path.exists(generator_checkpoint_path):
            checkpoint_g = torch.load(generator_checkpoint_path, map_location=self.device)
            if 'generator_state_dict' in checkpoint_g: self.generator.load_state_dict(checkpoint_g['generator_state_dict'])
            else: self.generator.load_state_dict(checkpoint_g) # Assume raw state_dict
            logger.info(f"Generator loaded from {generator_checkpoint_path}")
        else:
            logger.error(f"Generator checkpoint not found: {generator_checkpoint_path}. Real-time conversion will not work.")
            raise FileNotFoundError(f"Generator model not found at {generator_checkpoint_path}")
        self.generator.eval()

        # Load Vocoder
        logger.info("Loading HiFi-GAN Vocoder model...")
        vocoder_checkpoint = self.model_config['vocoder']['checkpoint_path']
        self.vocoder = HiFiGANVocoder(checkpoint_path=vocoder_checkpoint, config=full_config) # Pass full config
        if self.vocoder.model is None: # Check if actual model loaded in vocoder
            logger.warning("HiFi-GAN model in vocoder is None. Playback will be dummy audio.")

        # Noise suppression
        self.noise_suppressor = noisereduce if self.rt_config.get('noise_suppression_enabled', False) and noisereduce else None
        if self.noise_suppressor: logger.info("Noise suppression enabled.")
        else: logger.info("Noise suppression disabled or 'noisereduce' not available.")

        self.sample_rate = self.data_config['sample_rate']
        # Buffer size in samples (block_size for sounddevice stream)
        self.block_size_samples = self.rt_config.get('buffer_size_samples', int(0.1 * self.sample_rate)) # e.g. 100ms

        self.input_buffer = np.array([], dtype=np.float32)
        self.output_audio_buffer = collections.deque() # For queuing vocoded audio chunks

        # Determine target speaker ID
        if direction == 'male_to_female': self.target_speaker_id = self.rt_config.get('target_female_speaker_id',0)
        elif direction == 'female_to_male': self.target_speaker_id = self.rt_config.get('target_male_speaker_id',1)
        else: self.target_speaker_id = self.rt_config.get('specific_target_speaker_id',0)
        logger.info(f"Target speaker ID for conversion: {self.target_speaker_id}")

        # Pre-compute target speaker embedding (assuming Generator's MLP takes raw embedding)
        # This is a simplification. In practice, SpeakerAdaptationMLP is part of Generator.
        # Generator's forward takes speaker_embedding_target, which is output of SpeakerAdaptationMLP.
        # SpeakerAdaptationMLP itself takes an embedding from an nn.Embedding layer based on ID.
        # So we need an nn.Embedding layer here.
        temp_speaker_embedding_layer = nn.Embedding(
            self.model_config['num_speakers'],
            self.model_config['speaker_embedding_dim']
        ).to(self.device)
        self.target_speaker_embedding = temp_speaker_embedding_layer(
            torch.tensor([self.target_speaker_id], dtype=torch.long).to(self.device)
        )
        logger.info(f"Target speaker embedding prepared. Shape: {self.target_speaker_embedding.shape}")

        self.stream = None # Initialize stream attribute

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            logger.warning(f"Stream status: {status}")

        # Input audio processing (indata is NumPy array)
        input_chunk = indata[:, 0] # Assuming mono input, take first channel

        if self.noise_suppressor:
            try:
                # Ensure sample rate matches what noisereduce expects
                input_chunk = self.noise_suppressor.reduce_noise(y=input_chunk, sr=self.sample_rate, quiet=True)
            except Exception as e:
                logger.error(f"Error in noise reduction: {e}")

        # TODO: Implement proper streaming processing:
        # 1. Append input_chunk to self.input_buffer.
        # 2. If self.input_buffer has enough data for a mel-spectrogram frame (or window):
        #    a. Extract window from self.input_buffer (with potential overlap).
        #    b. Convert window to mel_spectrogram:
        #       mel_tensor = torch.from_numpy(wav_to_mel_spectrogram(...)).float().unsqueeze(0).to(self.device)
        #    c. Perform generator inference:
        #       with torch.no_grad(): output_mel = self.generator(mel_tensor, self.target_speaker_embedding)
        #    d. Vocode output_mel:
        #       vocoded_chunk = self.vocoder.mel_to_wav(output_mel.squeeze(0).cpu()) # Ensure correct shape and device
        #    e. Add vocoded_chunk (NumPy array) to self.output_audio_buffer.
        # 3. If self.output_audio_buffer has enough data for `outdata`:
        #    a. Pop data from self.output_audio_buffer and fill `outdata`.
        # 4. Else, fill `outdata` with zeros (silence).

        # For now, simple passthrough or silence for placeholder
        logger.debug(f"Audio callback: indata shape {indata.shape}, frames {frames}")

        # Passthrough for testing audio chain (mono)
        # outdata[:] = input_chunk.reshape(-1,1) if input_chunk.ndim == 1 else input_chunk

        # Silence if not enough output data
        if len(self.output_audio_buffer) >= frames:
            processed_chunk = self.output_audio_buffer.popleft() # This assumes chunks are already right size
            # This logic needs to be more robust: popleft might not be `frames` long.
            # A better way is to have a continuous buffer and read `frames` from it.
            # For now, let's assume it is, or fill with silence.
            outdata[:len(processed_chunk)] = processed_chunk.reshape(-1,1)
            if len(processed_chunk) < frames:
                outdata[len(processed_chunk):] = np.zeros((frames - len(processed_chunk), outdata.shape[1]), dtype=np.float32)
        else:
            # Fill with silence if not enough processed audio
            outdata[:] = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
            # Add a dummy processed chunk to simulate processing for testing output stream
            # This is where actual processed audio should go.
            # self.output_audio_buffer.append(np.zeros_like(indata[:,0])) # Add silence of input size


    def start(self):
        input_dev_idx = self.rt_config.get('input_device_index')
        output_dev_idx = self.rt_config.get('output_device_index')

        logger.info(f"Attempting to start real-time audio stream...")
        logger.info(f"  Input device ID: {input_dev_idx}, Output device ID: {output_dev_idx}")
        logger.info(f"  Sample rate: {self.sample_rate}, Block size (frames): {self.block_size_samples}")
        logger.info(f"  Channels: 1 (mono input/output assumed)")

        try:
            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size_samples, # Number of frames per callback
                device=(input_dev_idx, output_dev_idx),
                channels=1, # Mono
                dtype='float32', # Processing in float32
                callback=self._audio_callback,
                latency='low' # Request low latency
            )
            self.stream.start()
            logger.info("Audio stream started. Press Ctrl+C to stop.")
            # Keep the main thread alive while the stream is active in a background thread
            while self.stream and self.stream.active:
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}", exc_info=True)
            if self.stream:
                self.stream.close()
            self.stream = None


    def stop(self):
        logger.info("Stopping audio stream...")
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped and closed.")
        self.stream = None

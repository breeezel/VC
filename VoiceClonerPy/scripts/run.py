import argparse; import torch; import os; import logging; import time; import sys; import yaml
from src.utils.config_loader import load_config
from src.training.trainer import Trainer
from src.models.stargan_vc import Generator, Discriminator
from src.data_loader import VoiceDataset, DataLoader, prepare_base_model_data # Assuming prepare_base_model_data lists wav files
from src.inference import convert_voice_from_file
from src.realtime_audio_utils import list_audio_devices, select_device_id
from src.realtime_inference import RealTimeVoiceConverter

logger = logging.getLogger("VoiceClonerPy_RunScript"); logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout); fm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fm); logger.addHandler(ch)

def load_data_for_dataset(data_dir_or_file_list, metadata_file, num_speakers_from_model, is_validation=False):
    # Placeholder: This function needs to parse metadata_file or scan data_dir_or_file_list
    # to produce a list of tuples: [(audio_path, speaker_id)] for training,
    # or [(src_path, target_path, src_id, target_id)] for validation.
    # For now, returning dummy data.
    logger.warning(f"Data loading for {'validation' if is_validation else 'training'} is using DUMMY data. Implement actual data loading.")
    num_dummy = 20 if is_validation else 100

    if is_validation: # (source_mel_audio_path, target_eval_audio_path, source_speaker_id, target_speaker_id_for_conversion)
        return [(f"dummy_val_src_{i}.wav", f"dummy_val_trg_{i}.wav", i % num_speakers_from_model, (i+1) % num_speakers_from_model) for i in range(num_dummy)]
    else: # (audio_path, speaker_id)
        return [(f"dummy_train_{i}.wav", i % num_speakers_from_model) for i in range(num_dummy)]


def main(args):
    print("VoiceClonerPy starting..."); print(f"Config: {args.config}")
    try: config = load_config(args.config)
    except Exception as e: logger.error(f"Error loading config {args.config}: {e}. Exiting."); sys.exit(1)
    if not config: logger.error(f"Failed to load config (returned None) from {args.config}. Exiting."); sys.exit(1)

    effective_mode = config['training'].get('mode', 'train');
    if args.mode: effective_mode = args.mode; config['training']['mode'] = effective_mode
    print(f"Mode: {effective_mode}"); logger.info(f"Project: {config['project'].get('name')}, Exp: {config['project'].get('experiment_name')}, Mode: {effective_mode}")

    device = torch.device('cuda' if config['training'].get('device') == 'cuda' and torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    try:
        generator = Generator(config=config['model']).to(device)
        discriminator = Discriminator(num_speakers=config['model']['num_speakers'], **config['model']['discriminator']).to(device) # Pass discriminator sub-config
    except KeyError as e: logger.error(f"Model config error: {e}. Exiting."); sys.exit(1)
    except Exception as e: logger.error(f"Model init error: {e}. Exiting."); sys.exit(1)

    if effective_mode == 'train' or effective_mode == 'fine_tune': # Combined setup for train/fine-tune
        logger.info(f"Setting up for {effective_mode} mode...")

        data_dir = config['data']['fine_tune_user_data_dir'] if effective_mode == 'fine_tune' else config['data']['base_model_corpus_dir']
        metadata = config['data'].get('fine_tune_metadata', None) if effective_mode == 'fine_tune' else config['data'].get('base_model_metadata', None)
        # This is simplified data loading. Real implementation would use data_dir and metadata.
        train_data_entries = load_data_for_dataset(data_dir, metadata, config['model']['num_speakers'], is_validation=False)
        train_dataset = VoiceDataset(train_data_entries, config['data'], is_validation=False)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True,
                                  num_workers=config['data']['num_workers'], pin_memory=config['data']['pin_memory'])

        val_loader = None
        if config['data'].get('val_data_dir') and config['data'].get('val_metadata_file'):
            val_data_entries = load_data_for_dataset(
                config['data']['val_data_dir'],
                config['data']['val_metadata_file'],
                config['model']['num_speakers'],
                is_validation=True
            )
            if val_data_entries:
                val_dataset = VoiceDataset(val_data_entries, config['data'], is_validation=True)
                val_loader = DataLoader(val_dataset, batch_size=config['training'].get('batch_size_val', 4), shuffle=False,
                                        num_workers=config['data']['num_workers'], pin_memory=config['data']['pin_memory'])
                logger.info(f"Validation dataset loaded with {len(val_dataset)} samples.")
            else: logger.warning("Validation data entries could not be loaded.")
        else: logger.info("No validation data directory or metadata file specified. Skipping validation.")

        if effective_mode == 'fine_tune':
            ft_conf = config.get('fine_tuning', {})
            config['training']['num_epochs'] = ft_conf.get('num_epochs', config['training']['num_epochs'])
            # ... (other ft overrides) ...
            base_model_g_path = ft_conf.get('base_model_checkpoint_path')
            if base_model_g_path and os.path.exists(base_model_g_path):
                logger.info(f"Loading base G from: {base_model_g_path} for fine-tuning.")
                try:
                    gen_ckpt = torch.load(base_model_g_path, map_location=device)
                    if 'generator_state_dict' in gen_ckpt: generator.load_state_dict(gen_ckpt['generator_state_dict'])
                    else: generator.load_state_dict(gen_ckpt) # Assume raw state dict
                except Exception as e: logger.error(f"Error loading base G for FT: {e}")
            else: logger.warning(f"Base G for FT not found: {base_model_g_path}")

        trainer = Trainer(config, generator, discriminator, train_loader, val_loader, device)
        logger.info(f"Trainer for {effective_mode} initialized. Starting...")
        trainer.run_training()
        logger.info(f"{effective_mode.capitalize()} finished.")

    elif effective_mode == 'inference_file': # (Same as previous version, check for consistency)
        logger.info("Setting up for inference_file mode...")
        inf_conf = config.get('inference_file');
        if not inf_conf: logger.error("`inference_file` section missing. Exiting."); sys.exit(1)
        # ... (rest of inference_file logic as before)
        model_path_to_load, target_speaker_id = None, 0
        direction = inf_conf.get('conversion_direction', 'specific')
        if direction == 'male_to_female': model_path_to_load,target_speaker_id = inf_conf.get('fine_tuned_model_path_female_generator'),inf_conf.get('target_female_speaker_id',0)
        elif direction == 'female_to_male': model_path_to_load,target_speaker_id = inf_conf.get('fine_tuned_model_path_male_generator'),inf_conf.get('target_male_speaker_id',1)
        elif direction == 'specific': model_path_to_load,target_speaker_id = inf_conf.get('specific_generator_checkpoint_path'),inf_conf.get('specific_target_speaker_id',0)
        else: logger.error(f"Invalid conversion_direction: {direction}. Exiting."); sys.exit(1)
        if not model_path_to_load or not os.path.exists(model_path_to_load): logger.error(f"Model for inference not found: {model_path_to_load}. Exiting."); sys.exit(1)
        convert_voice_from_file(config, model_path_to_load, inf_conf['input_wav_path'], inf_conf['output_wav_path'], target_speaker_id)

    elif effective_mode == 'inference_realtime': # (Same as previous version, check for consistency)
        logger.info("Setting up for inference_realtime mode...")
        rt_conf = config.get('inference_realtime');
        if not rt_conf: logger.error("`inference_realtime` section missing. Exiting."); sys.exit(1)
        list_audio_devices()
        # ... (device selection logic as before) ...
        input_dev_idx = rt_conf.get('input_device_index', -1)
        output_dev_idx = rt_conf.get('output_device_index', -1)
        if input_dev_idx == -1 : input_dev_idx = select_device_id("Select INPUT device ID: ", kind="input"); rt_conf['input_device_index'] = input_dev_idx
        if output_dev_idx == -1 : output_dev_idx = select_device_id("Select OUTPUT device ID: ", kind="output"); rt_conf['output_device_index'] = output_dev_idx
        if input_dev_idx is None or output_dev_idx is None: logger.error("Device selection cancelled or invalid. Exiting."); sys.exit(1)
        try:
            converter = RealTimeVoiceConverter(config); converter.start()
        except KeyboardInterrupt: logger.info("Real-time conversion stopped by user.")
        except Exception as e: logger.error(f"Error during real-time inference: {e}", exc_info=True)
        finally:
            if 'converter' in locals() and hasattr(converter, 'stop'): converter.stop()
        logger.info("Real-time inference session ended.")

    else: logger.error(f"Unknown mode: '{effective_mode}'. Exiting."); sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VoiceClonerPy Operations.")
    parser.add_argument('--config',type=str,default="config/config_template.yaml",help="Path to YAML config.")
    parser.add_argument('--mode',type=str,choices=['train','fine_tune','inference_file','inference_realtime'],help="Override mode.")
    cmd_args = parser.parse_args()
    main(cmd_args)

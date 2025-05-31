import sounddevice as sd
import numpy as np

def list_audio_devices():
    """Lists available audio input and output devices."""
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"  ID {i}: {device['name']}")
        print(f"    Max input channels: {device['max_input_channels']}")
        print(f"    Max output channels: {device['max_output_channels']}")
        print(f"    Default samplerate: {device['default_samplerate']}")
    return devices

def select_device_id(prompt_message="Select device by ID: ", kind="input/output"):
    """
    Prompts the user to select an audio device ID from the listed devices.
    Args:
        prompt_message (str): The message to display to the user.
        kind (str): Type of device being selected, for context in prompt.
    Returns:
        int: The selected device ID, or None if input is invalid.
    """
    while True:
        try:
            device_id_str = input(prompt_message)
            if not device_id_str: # Handle empty input, perhaps by returning a default or None
                print(f"No device selected for {kind}. Please provide a valid ID or check your configuration.")
                return None
            device_id = int(device_id_str)
            # Basic validation, assumes sd.query_devices() would have been called and shown to user
            # A more robust version would check against len(sd.query_devices())
            if device_id >= 0:
                return device_id
            else:
                print("Invalid device ID. Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter a numerical ID.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

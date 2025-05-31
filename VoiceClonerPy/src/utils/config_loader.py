import yaml
import os

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration as a dictionary, or None if an error occurs.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        # Consider raising FileNotFoundError instead of returning None for clearer error handling upstream
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return config_dict
    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading config {config_path}: {e}")
        return None

def save_config(config_dict, config_path):
    """
    Saves a dictionary to a YAML configuration file.

    Args:
        config_dict (dict): Configuration dictionary to save.
        config_path (str): Path to save the YAML configuration file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}")
        return False

# utils/config_parser.py

"""
Parses a YAML configuration file.
"""

import yaml
from pathlib import Path

def parse_config(config_path: str | Path) -> dict:
    """
    Parses a YAML configuration file.

    Args:
        config_path (str or Path): The path to the YAML config file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        raise
    except Exception as e:
        print(f"Error parsing the YAML file: {e}")
        raise
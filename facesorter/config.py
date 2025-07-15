import yaml
import os

# --- Constants ---
# Define common directories as constants to be used across the application.
OUTPUT_DIR = "sorted_output"
TEMP_UPLOAD_DIR = "temp_uploads"
TEMP_CROP_DIR = "temp_crops"
BATCH_SIZE = 100 # Default batch size

class Config:
    """A class to manage configuration settings for the FaceSorter application."""

    def __init__(self, config_file="config.yaml"):
        """
        Initializes the Config object by loading settings from a YAML file.

        Args:
            config_file (str): The path to the configuration file.
        """
        with open(config_file, 'r') as f:
            self.settings = yaml.safe_load(f)

    def get(self, key, default=None):
        """
        Retrieves a configuration value for a given key.

        Args:
            key (str): The configuration key to retrieve, using dot notation for nested keys.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value or the default value.
        """
        keys = key.split('.')
        value = self.settings
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# A global instance of the Config class to be used throughout the application
config = Config() 
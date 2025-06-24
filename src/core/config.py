"""
Configuration management for the Mindstream project.
Handles all settings, environment variables, and configuration loading.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Central configuration management for the Mindstream project."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/settings.yaml"
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Default configuration
        self._config = {
            'audio': {
                'base_download_dir': 'temp_audio',
                'max_concurrent': 2,
                'max_retries': 3,
                'sample_rate': 22050,
                'hop_length': 512,
                'min_duration': 1.0,
                'max_duration': 600.0,
                'min_file_size': 10000,
                'vocal_band_low': 300,
                'vocal_band_high': 3400,
                'vocal_activity_threshold': 0.15,
                'voiced_ratio_threshold': 0.2,
                'pitch_variance_threshold': 100,
                'rms_variance_threshold': 0.02,
                'energy_threshold': 0.3
            },
            'database': {
                'type': 'supabase',  # 'supabase' or 'sqlite'
                'url': os.getenv('SUPABASE_URL'),
                'key': os.getenv('SUPABASE_KEY'),
                'sqlite_path': 'data/tracks.db'
            },
            'spotify': {
                'client_id': os.getenv('SPOTIFY_CLIENT_ID'),
                'client_secret': os.getenv('SPOTABASE_CLIENT_SECRET'),
                'redirect_uri': os.getenv('SPOTIFY_REDIRECT_URI'),
                'token_file': 'data/spotify_token.json'
            },
            'tracker': {
                'polling_interval': 5,
                'token_refresh_interval': 3300,  # 55 minutes
                'skip_short_plays': True,
                'min_play_duration': 30,
                'batch_size': 10,
                'max_batch_workers': 3
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/mindstream.log'
            }
        }
        
        # Load from YAML file if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                self._merge_config(file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_path}: {e}")
        
        # Override with environment variables
        self._load_env_overrides()
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Recursively merge new configuration into existing config."""
        for key, value in new_config.items():
            if key in self._config and isinstance(self._config[key], dict) and isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'AUDIO_MAX_CONCURRENT': ('audio', 'max_concurrent', int),
            'AUDIO_MAX_RETRIES': ('audio', 'max_retries', int),
            'AUDIO_SAMPLE_RATE': ('audio', 'sample_rate', int),
            'DATABASE_TYPE': ('database', 'type', str),
            'SPOTIFY_CLIENT_ID': ('spotify', 'client_id', str),
            'SPOTIFY_CLIENT_SECRET': ('spotify', 'client_secret', str),
            'TRACKER_POLLING_INTERVAL': ('tracker', 'polling_interval', int),
            'TRACKER_TOKEN_REFRESH_INTERVAL': ('tracker', 'token_refresh_interval', int),
            'LOG_LEVEL': ('logging', 'level', str),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    self._config[section][key] = type_func(value)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid value for {env_var}: {value}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        return self._config.get(section, {})
    
    def set(self, section: str, key: str, value: Any):
        """Set a configuration value."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def save(self, config_path: Optional[str] = None):
        """Save current configuration to file."""
        path = config_path or self.config_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def validate(self) -> bool:
        """Validate that all required configuration is present."""
        required_configs = [
            ('spotify', 'client_id'),
            ('spotify', 'client_secret'),
            ('database', 'url'),
            ('database', 'key'),
        ]
        
        missing = []
        for section, key in required_configs:
            if not self.get(section, key):
                missing.append(f"{section}.{key}")
        
        if missing:
            print(f"Missing required configuration: {', '.join(missing)}")
            return False
        
        return True


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config 
"""
Audio Validator Module for Mindstream
Validates audio files and analysis results.
"""

import os
import logging
from typing import Dict, Any, Optional
import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioValidator:
    """Validates audio files and analysis results."""
    
    def __init__(self, config):
        """
        Initialize the audio validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.min_duration = config.get('audio', 'min_duration', 1.0)
        self.max_duration = config.get('audio', 'max_duration', 600.0)
        self.min_sample_rate = config.get('audio', 'min_sample_rate', 8000)
        self.max_sample_rate = config.get('audio', 'max_sample_rate', 48000)
        
        logger.info("✅ Audio Validator initialized")
    
    def is_valid_audio_file(self, file_path: str) -> bool:
        """
        Validate if an audio file is suitable for analysis.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if the file is valid, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Audio file does not exist: {file_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            min_size = 1024  # 1KB minimum
            if file_size < min_size:
                logger.error(f"Audio file too small ({file_size} bytes): {file_path}")
                return False
            
            # Load audio to check duration and sample rate
            y, sr = librosa.load(file_path, sr=None, mono=True)
            
            # Check sample rate
            if sr < self.min_sample_rate or sr > self.max_sample_rate:
                logger.error(f"Invalid sample rate {sr}Hz: {file_path}")
                return False
            
            # Check duration
            duration = len(y) / sr
            if duration < self.min_duration:
                logger.error(f"Audio too short ({duration:.2f}s): {file_path}")
                return False
            if duration > self.max_duration:
                logger.warning(f"Audio very long ({duration:.2f}s), will analyze first 5 minutes: {file_path}")
            
            # Check for silence or very low amplitude
            rms = librosa.feature.rms(y=y)[0]
            if np.mean(rms) < 0.001:
                logger.error(f"Audio too quiet (silent or near-silent): {file_path}")
                return False
            
            logger.info(f"Audio file validated successfully: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {e}")
            return False
    
    def validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate analysis features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            True if features are valid, False otherwise
        """
        try:
            required_features = [
                'tempo', 'valence', 'danceability', 'instrumentalness', 
                'acousticness', 'speechiness', 'key', 'mode', 'energy', 'loudness'
            ]
            
            # Check for required features
            for feature in required_features:
                if feature not in features:
                    logger.error(f"Missing required feature: {feature}")
                    return False
            
            # Validate feature ranges
            validations = [
                ('tempo', 0, 300),
                ('valence', 0, 1),
                ('danceability', 0, 1),
                ('instrumentalness', 0, 1),
                ('acousticness', 0, 1),
                ('speechiness', 0, 1),
                ('energy', 0, 1),
                ('loudness', -60, 0)
            ]
            
            for feature, min_val, max_val in validations:
                value = features.get(feature)
                if value is None:
                    logger.error(f"Feature {feature} is None")
                    return False
                if not isinstance(value, (int, float)):
                    logger.error(f"Feature {feature} is not numeric: {type(value)}")
                    return False
                if value < min_val or value > max_val:
                    logger.warning(f"Feature {feature} ({value}) outside expected range [{min_val}, {max_val}]")
            
            # Validate key and mode
            valid_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'Unknown']
            valid_modes = ['major', 'minor', 'Unknown']
            
            if features.get('key') not in valid_keys:
                logger.warning(f"Invalid key: {features.get('key')}")
            
            if features.get('mode') not in valid_modes:
                logger.warning(f"Invalid mode: {features.get('mode')}")
            
            logger.info("Features validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating features: {e}")
            return False
    
    def validate_analysis_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate a complete analysis result.
        
        Args:
            result: Complete analysis result dictionary
            
        Returns:
            True if result is valid, False otherwise
        """
        try:
            # Check for required metadata
            required_metadata = ['track_id', 'name', 'artist', 'audio_file_path']
            for field in required_metadata:
                if field not in result:
                    logger.error(f"Missing required metadata: {field}")
                    return False
            
            # Validate audio file path
            if not self.is_valid_audio_file(result['audio_file_path']):
                return False
            
            # Extract features for validation
            features = {k: v for k, v in result.items() 
                       if k not in ['track_id', 'name', 'artist', 'album', 'duration_ms', 
                                   'popularity', 'audio_file_path', 'analysis_timestamp']}
            
            # Validate features
            if not self.validate_features(features):
                return False
            
            logger.info(f"Analysis result validation passed for: {result.get('name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating analysis result: {e}")
            return False 
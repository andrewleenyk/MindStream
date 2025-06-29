"""
Mozart Enhanced Audio Analyzer for Mindstream
Integrates the Mozart enhanced audio analysis algorithms with Mindstream's database system.
"""

import os
import logging
import warnings
from typing import Dict, Any, Optional, Tuple
import numpy as np
import librosa
import soundfile as sf
from scipy import stats
from scipy.signal import correlate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Fix for scipy.signal.hann compatibility with newer SciPy versions
try:
    from scipy.signal import hann
except ImportError:
    try:
        from scipy.signal.windows import hann
        # Monkey patch scipy.signal to include hann for librosa compatibility
        import scipy.signal
        scipy.signal.hann = hann
    except ImportError:
        # Fallback: create hann window manually and patch it
        def hann(M):
            return 0.5 * (1 - np.cos(2 * np.pi * np.arange(M) / (M - 1)))
        import scipy.signal
        scipy.signal.hann = hann

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MozartAudioAnalyzer:
    """Enhanced audio analyzer using Mozart algorithms, integrated with Mindstream."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the Mozart enhanced analyzer.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        logger.info("✅ Mozart Enhanced Audio Analyzer initialized")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to mono.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(file_path, mono=True, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_enhanced_valence(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract valence using advanced spectral and temporal analysis.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (valence_score, confidence)
        """
        try:
            # Spectral features for valence
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            # Zero crossing rate (noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Chroma features for harmonic content
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
            
            # Calculate valence indicators
            # Higher spectral centroid often correlates with positive valence
            brightness_score = np.mean(spectral_centroid) / 4000.0
            
            # Lower zero crossing rate often indicates less noise (more positive)
            noise_score = 1.0 - np.clip(np.mean(zcr), 0.0, 0.5) / 0.5
            
            # Harmonic content (major vs minor characteristics)
            chroma_mean = np.mean(chroma, axis=1)
            major_profile = np.array([1, 0, 0.5, 0, 0.8, 0.3, 0, 1, 0, 0.5, 0, 0.8])
            minor_profile = np.array([1, 0, 0.5, 0.8, 0, 0.3, 0, 1, 0, 0.5, 0.8, 0])
            
            major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
            harmonic_score = max(0, major_corr - minor_corr)
            
            # Combine scores
            valence_score = (brightness_score + noise_score + harmonic_score) / 3.0
            valence_score = np.clip(valence_score, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([brightness_score, noise_score, harmonic_score])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(valence_score), float(confidence)
            
        except Exception as e:
            logger.error(f"Error extracting valence: {e}")
            return 0.5, 0.0
    
    def extract_enhanced_danceability(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract danceability using advanced rhythmic analysis.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (danceability_score, confidence)
        """
        try:
            # Tempo analysis
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Onset strength
            onset_strength = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Calculate danceability indicators
            # Tempo score (optimal range 120-140 BPM)
            tempo_score = 0.0
            if 120 <= tempo <= 140:
                tempo_score = 1.0
            elif 100 <= tempo <= 160:
                tempo_score = 0.8
            elif 80 <= tempo <= 180:
                tempo_score = 0.6
            else:
                tempo_score = 0.3
            
            # Beat strength
            beat_strength = np.mean(onset_strength)
            strength_score = np.clip(beat_strength / 10.0, 0.0, 1.0)
            
            # Beat regularity
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                regularity = 1.0 - np.clip(np.std(beat_intervals) / np.mean(beat_intervals), 0.0, 1.0)
            else:
                regularity = 0.0
            
            # Spectral flux (energy changes)
            spectral_flux = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=512)
            flux_score = np.clip(np.mean(spectral_flux) / 5.0, 0.0, 1.0)
            
            # Combine scores
            danceability = (tempo_score + strength_score + regularity + flux_score) / 4.0
            danceability = np.clip(danceability, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([tempo_score, strength_score, regularity, flux_score])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(danceability), float(confidence)
            
        except Exception as e:
            logger.error(f"Error extracting danceability: {e}")
            return 0.5, 0.0
    
    def extract_enhanced_instrumentalness(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract instrumentalness using harmonic/percussive separation.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (instrumentalness_score, confidence)
        """
        try:
            # Harmonic/percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Calculate energy ratios
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy > 0:
                harmonic_ratio = harmonic_energy / total_energy
            else:
                harmonic_ratio = 0.5
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # Vocal activity detection (simplified)
            # Higher spectral centroid in vocal range (300-3400 Hz) suggests vocals
            vocal_energy = np.mean(spectral_centroid[spectral_centroid > 300])
            vocal_score = 1.0 - np.clip(vocal_energy / 4000.0, 0.0, 1.0)
            
            # Spectral flatness (noise vs harmonic content)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            flatness_score = np.mean(spectral_flatness)
            
            # Combine scores
            instrumentalness = (harmonic_ratio + vocal_score + flatness_score) / 3.0
            instrumentalness = np.clip(instrumentalness, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([harmonic_ratio, vocal_score, flatness_score])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(instrumentalness), float(confidence)
            
        except Exception as e:
            logger.error(f"Error extracting instrumentalness: {e}")
            return 0.5, 0.0
    
    def extract_enhanced_acousticness(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract acousticness using spectral analysis.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (acousticness_score, confidence)
        """
        try:
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            # Calculate acousticness indicators
            # Lower spectral centroid often indicates acoustic instruments
            centroid_score = 1.0 - np.clip(np.mean(spectral_centroid) / 4000.0, 0.0, 1.0)
            
            # Lower spectral rolloff often indicates acoustic instruments
            rolloff_score = 1.0 - np.clip(np.mean(spectral_rolloff) / 8000.0, 0.0, 1.0)
            
            # Spectral bandwidth (acoustic instruments often have narrower bandwidth)
            bandwidth_score = 1.0 - np.clip(np.mean(spectral_bandwidth) / 4000.0, 0.0, 1.0)
            
            # MFCC variance (acoustic instruments often have more consistent timbre)
            mfcc_var = np.var(mfcc, axis=1)
            mfcc_score = 1.0 - np.clip(np.mean(mfcc_var) / 10.0, 0.0, 1.0)
            
            # Combine scores
            acousticness = (centroid_score + rolloff_score + bandwidth_score + mfcc_score) / 4.0
            acousticness = np.clip(acousticness, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([centroid_score, rolloff_score, bandwidth_score, mfcc_score])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(acousticness), float(confidence)
            
        except Exception as e:
            logger.error(f"Error extracting acousticness: {e}")
            return 0.5, 0.0
    
    def extract_enhanced_speechiness(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract speechiness using spectral and temporal analysis.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (speechiness_score, confidence)
        """
        try:
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            # Calculate speechiness indicators
            # High spectral centroid variance indicates speech
            centroid_var = np.var(spectral_centroid)
            centroid_score = np.clip(centroid_var / 1000000.0, 0.0, 1.0)
            
            # High zero crossing rate indicates speech
            zcr_score = np.clip(np.mean(zcr) / 0.1, 0.0, 1.0)
            
            # High spectral rolloff variance indicates speech
            rolloff_var = np.var(spectral_rolloff)
            rolloff_score = np.clip(rolloff_var / 10000000.0, 0.0, 1.0)
            
            # MFCC variance (speech has more variable timbre)
            mfcc_var = np.var(mfcc, axis=1)
            mfcc_score = np.clip(np.mean(mfcc_var) / 20.0, 0.0, 1.0)
            
            # Combine scores
            speechiness = (centroid_score + zcr_score + rolloff_score + mfcc_score) / 4.0
            speechiness = np.clip(speechiness, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([centroid_score, zcr_score, rolloff_score, mfcc_score])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(speechiness), float(confidence)
            
        except Exception as e:
            logger.error(f"Error extracting speechiness: {e}")
            return 0.5, 0.0
    
    def extract_all_features(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract the 10 enhanced features (and their confidence scores) plus key, mode, energy, and loudness using the Mozart repo logic.
        """
        try:
            # Load audio
            audio, sr = self.load_audio(audio_file_path)
            if audio is None:
                return None

            # Extract enhanced features using Mozart logic
            valence, valence_conf = self.extract_enhanced_valence(audio)
            danceability, dance_conf = self.extract_enhanced_danceability(audio)
            instrumentalness, inst_conf = self.extract_enhanced_instrumentalness(audio)
            acousticness, acous_conf = self.extract_enhanced_acousticness(audio)
            speechiness, speech_conf = self.extract_enhanced_speechiness(audio)

            # Key and mode detection (using chroma)
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)

            try:
                # For newer versions of librosa
                key_raw, mode_raw = librosa.feature.key_mode(chroma)
            except AttributeError:
                # Fallback for older versions
                try:
                    chroma_sum = np.sum(chroma, axis=1)
                    key_raw = np.argmax(chroma_sum)
                    mode_raw = 0  # Default to major
                except:
                    key_raw = 0
                    mode_raw = 0

            # Convert to strings
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            mode_names = ['major', 'minor']

            key = key_names[key_raw] if key_raw < len(key_names) else 'Unknown'
            mode = mode_names[mode_raw] if mode_raw < len(mode_names) else 'Unknown'

            # Energy and loudness
            rms = librosa.feature.rms(y=audio)[0]
            energy = float(np.mean(rms))
            loudness = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))

            features = {
                'valence': float(valence),
                'valence_confidence': float(valence_conf),
                'danceability': float(danceability),
                'danceability_confidence': float(dance_conf),
                'instrumentalness': float(instrumentalness),
                'instrumentalness_confidence': float(inst_conf),
                'acousticness': float(acousticness),
                'acousticness_confidence': float(acous_conf),
                'speechiness': float(speechiness),
                'speechiness_confidence': float(speech_conf),
                'key': key,
                'mode': mode,
                'energy': energy,
                'loudness': loudness
            }
            return features
        except Exception as e:
            logger.error(f"Failed to extract Mozart features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None 
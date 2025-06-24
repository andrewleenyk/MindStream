"""
Audio feature extraction module.
Contains all the logic for extracting musical features from audio files.
"""

import numpy as np
import librosa
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extracts musical features from audio using librosa."""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
    
    def extract_all_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Extract all audio features from the audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary containing all extracted features
        """
        try:
            # Basic audio preprocessing
            y = librosa.util.normalize(y)
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Check if audio is too quiet after trimming
            if np.max(np.abs(y)) < 0.01:
                logger.warning("Audio too quiet after trimming")
                return None
            
            # Extract rhythmic features
            tempo, beat_frames, beat_intervals = self._extract_rhythmic_features(y, sr)
            
            # Extract spectral features
            spectral_features = self._extract_spectral_features(y, sr)
            
            # Extract harmonic features
            harmonic_features = self._extract_harmonic_features(y, sr)
            
            # Extract pitch features
            pitch_features = self._extract_pitch_features(y, sr)
            
            # Extract onset features
            onset_features = self._extract_onset_features(y, sr)
            
            # Extract key and mode
            key, mode = self._extract_key_mode(y, sr)
            
            # Calculate derived features
            derived_features = self._calculate_derived_features(
                tempo, beat_frames, beat_intervals, spectral_features, 
                harmonic_features, pitch_features, onset_features, key, mode
            )
            
            # Compile all features
            features = {
                'tempo': round(float(np.clip(tempo, 40, 200)), 2),
                'beat_strength': round(float(np.clip(derived_features['beat_strength'], 0, 1)), 2),
                'rhythmic_stability': round(float(np.clip(derived_features['rhythmic_stability'], 0, 1)), 2),
                'regularity': round(float(np.clip(derived_features['regularity'], 0, 1)), 2),
                'valence': round(float(np.clip(derived_features['valence'], 0, 1)), 2),
                'key': key,
                'mode': mode,
                'energy': round(float(np.clip(derived_features['energy'], 0, 1)), 2),
                'loudness': round(float(np.clip(derived_features['loudness'], -60, 0)), 2),
                'instrumentalness': round(float(np.clip(derived_features['instrumentalness'], 0, 1)), 2),
                'acousticness': round(float(np.clip(derived_features['acousticness'], 0, 1)), 2),
                'speechiness': round(float(np.clip(derived_features['speechiness'], 0, 1)), 2),
                'danceability': round(float(np.clip(derived_features['danceability'], 0, 1)), 2)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return None
    
    def _extract_rhythmic_features(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """Extract tempo, beat frames, and beat intervals."""
        # Extract tempo and beat frames with enhanced parameters
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, 
            hop_length=self.hop_length,
            start_bpm=120.0,
            tightness=100
        )
        
        # Validate tempo detection
        if tempo < 40 or tempo > 200:
            logger.warning(f"Unusual tempo detected: {tempo}, using fallback")
            tempo = 120.0
        
        # Extract beat intervals
        if len(beat_frames) > 1:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)
            beat_intervals = np.diff(beat_times)
        else:
            beat_intervals = np.array([0.5])  # Default interval
        
        return tempo, beat_frames, beat_intervals
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features."""
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        
        return {
            'rms': rms,
            'rms_mean': float(np.mean(rms)),
            'rms_var': float(np.var(rms)),
            'spectral_centroids': spectral_centroids,
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_var': float(np.var(spectral_centroids)),
            'spectral_rolloff': spectral_rolloff,
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_var': float(np.var(spectral_rolloff)),
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_flatness': spectral_flatness,
            'spectral_flatness_mean': float(np.mean(spectral_flatness)),
            'spectral_contrast': spectral_contrast,
            'spectral_contrast_mean': float(np.mean(spectral_contrast)),
            'zcr': zcr,
            'zcr_mean': float(np.mean(zcr))
        }
    
    def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract harmonic and percussive features."""
        # Harmonic/percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
        
        # Calculate energies
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        total_energy = harmonic_energy + percussive_energy + 1e-6
        
        # Vocal activity detection
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        vocal_band = (freqs > 300) & (freqs < 3400)
        vocal_energy = np.mean(S[vocal_band, :])
        total_energy_stft = np.mean(S)
        vocal_activity_ratio = vocal_energy / (total_energy_stft + 1e-6)
        
        return {
            'harmonic_energy': harmonic_energy,
            'percussive_energy': percussive_energy,
            'total_energy': total_energy,
            'harmonic_ratio': harmonic_energy / total_energy,
            'harmonic_percussive_ratio': harmonic_energy / (percussive_energy + 1e-6),
            'vocal_activity_ratio': vocal_activity_ratio
        }
    
    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract pitch-related features."""
        try:
            pitches, voiced_flags, _ = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7')
            )
            voiced_ratio = np.mean(voiced_flags.astype(float))
            pitch_var = np.nanvar(pitches)
        except Exception:
            voiced_ratio = 0.0
            pitch_var = 0.0
        
        return {
            'voiced_ratio': voiced_ratio,
            'pitch_var': pitch_var
        }
    
    def _extract_onset_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract onset-related features."""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_env_var = np.var(onset_env)
        strong_onset_ratio = np.mean(onset_env > np.percentile(onset_env, 75))
        
        return {
            'onset_env': onset_env,
            'onset_env_var': onset_env_var,
            'strong_onset_ratio': strong_onset_ratio
        }
    
    def _extract_key_mode(self, y: np.ndarray, sr: int) -> Tuple[str, str]:
        """Extract key and mode."""
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
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
        
        return key, mode
    
    def _calculate_derived_features(self, tempo: float, beat_frames: np.ndarray, 
                                  beat_intervals: np.ndarray, spectral_features: Dict[str, float],
                                  harmonic_features: Dict[str, float], pitch_features: Dict[str, float],
                                  onset_features: Dict[str, float], key: str, mode: str) -> Dict[str, float]:
        """Calculate derived features from extracted features."""
        
        # Beat strength
        rms = spectral_features['rms']
        if len(beat_frames) > 0 and max(beat_frames) < len(rms):
            beat_strength = np.mean(rms[beat_frames])
        else:
            beat_strength = spectral_features['rms_mean']
        
        # Rhythmic stability
        if len(beat_intervals) > 1:
            rhythmic_stability = 1.0 / (1.0 + np.std(beat_intervals))
        else:
            rhythmic_stability = 0.5
        
        # Regularity
        if len(beat_intervals) > 1:
            autocorr = np.correlate(beat_intervals, beat_intervals, mode='full')
            regularity = np.max(autocorr[len(autocorr)//2:]) / len(beat_intervals)
        else:
            regularity = 0.0
        
        # Energy and loudness
        energy = spectral_features['rms_mean']
        loudness = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))
        
        # Valence (improved)
        is_minor = (mode == 'minor')
        valence = float(np.clip(
            (harmonic_features['harmonic_ratio'] * 0.2 +
             spectral_features['spectral_contrast_mean'] * 0.15 +
             (spectral_features['spectral_centroid_mean'] / self.sample_rate) * 0.15 +
             (spectral_features['spectral_rolloff_mean'] / self.sample_rate) * 0.1 +
             (0.2 if not is_minor else 0.0) +
             (0.2 if energy > 0.3 else 0.0) -
             (0.2 if spectral_features['rms_var'] > 0.02 else 0.0)), 0.0, 1.0
        ))
        
        # Instrumentalness (improved)
        instrumentalness = float(np.clip(
            (harmonic_features['harmonic_percussive_ratio'] * 0.4 +
             spectral_features['spectral_flatness_mean'] * 0.2 +
             (spectral_features['spectral_bandwidth_mean'] / self.sample_rate) * 0.1 +
             (0.3 if harmonic_features['vocal_activity_ratio'] < 0.15 else 0.0)), 0.0, 1.0
        ))
        
        # Acousticness
        acousticness = float(np.clip(
            (harmonic_features['harmonic_ratio'] * 0.6 + 
             (1.0 - spectral_features['spectral_flatness_mean']) * 0.4), 0.0, 1.0
        ))
        
        # Speechiness (improved)
        speechiness = float(np.clip(
            (spectral_features['spectral_centroid_var'] * 0.2 +
             spectral_features['zcr_mean'] * 0.2 +
             spectral_features['spectral_rolloff_var'] * 0.2 +
             (0.4 if (pitch_features['voiced_ratio'] < 0.2 and pitch_features['pitch_var'] < 100) else 0.0)), 0.0, 1.0
        ))
        
        # Danceability (improved)
        tempo_scalar = float(tempo) if hasattr(tempo, '__len__') else tempo
        tempo_factor = float(np.clip(tempo_scalar / 120.0, 0.5, 2.0))
        
        if len(beat_intervals) > 1:
            beat_regularity = 1.0 / (1.0 + np.std(beat_intervals))
        else:
            beat_regularity = 0.5
        
        rhythm_strength = beat_strength * rhythmic_stability
        
        danceability = float(np.clip(
            (tempo_factor * 0.2 +
             beat_regularity * 0.2 +
             rhythm_strength * 0.2 +
             onset_features['strong_onset_ratio'] * 0.2 -
             (0.2 if spectral_features['rms_var'] > 0.02 else 0.0)), 0.0, 1.0
        ))
        
        return {
            'beat_strength': beat_strength,
            'rhythmic_stability': rhythmic_stability,
            'regularity': regularity,
            'energy': energy,
            'loudness': loudness,
            'valence': valence,
            'instrumentalness': instrumentalness,
            'acousticness': acousticness,
            'speechiness': speechiness,
            'danceability': danceability
        } 
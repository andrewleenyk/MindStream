import yt_dlp
import librosa
import os
import tempfile
import logging
from typing import Dict, Any, Optional
import numpy as np
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import hashlib

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Downloads tracks and analyzes audio features using librosa with thread pooling and retries."""
    
    def __init__(self, base_download_dir: str = "temp_audio", max_concurrent: int = 2, max_retries: int = 3):
        """Initialize the audio analyzer with thread pool and caching."""
        self.base_download_dir = base_download_dir
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        
        # Thread pool for concurrent analysis
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        # In-memory cache for analyzed tracks
        self.analyzed_tracks = set()
        self.cache_lock = Lock()  # Thread-safe cache access
        
        # Analysis statistics
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cached_hits': 0,
            'download_failures': 0,
            'analysis_failures': 0,
            'validation_failures': 0,
            'total_duration': 0.0,
            'average_duration': 0.0
        }
        
        # Ensure base directory exists
        self.ensure_base_download_dir()
        
        logger.info(f"AudioAnalyzer initialized with max_concurrent={max_concurrent}, max_retries={max_retries}")
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    def ensure_base_download_dir(self):
        """Create base download directory if it doesn't exist."""
        if not os.path.exists(self.base_download_dir):
            os.makedirs(self.base_download_dir)
            logger.info(f"Created base download directory: {self.base_download_dir}")
    
    def create_unique_download_dir(self) -> str:
        """Create a unique temporary directory for this analysis."""
        unique_id = str(uuid.uuid4())[:8]
        download_dir = os.path.join(self.base_download_dir, f"analysis_{unique_id}")
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"Created unique download directory: {download_dir}")
        return download_dir
    
    def get_track_hash(self, track_name: str, artist_name: str) -> str:
        """Create a hash for the track to use as cache key."""
        track_string = f"{track_name.lower()}_{artist_name.lower()}"
        return hashlib.md5(track_string.encode()).hexdigest()[:8]
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get current analysis statistics."""
        stats = self.stats.copy()
        if stats['total_analyses'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / stats['total_analyses']
            stats['cache_hit_rate'] = stats['cached_hits'] / stats['total_analyses']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        return stats

    def reset_stats(self):
        """Reset analysis statistics."""
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cached_hits': 0,
            'download_failures': 0,
            'analysis_failures': 0,
            'validation_failures': 0,
            'total_duration': 0.0,
            'average_duration': 0.0
        }
        logger.info("Analysis statistics reset")

    def is_track_analyzed(self, track_name: str, artist_name: str) -> bool:
        """Check if a track has been analyzed (thread-safe)."""
        track_hash = self.get_track_hash(track_name, artist_name)
        with self.cache_lock:
            return track_hash in self.analyzed_tracks

    def mark_track_analyzed(self, track_name: str, artist_name: str):
        """Mark a track as analyzed (thread-safe)."""
        track_hash = self.get_track_hash(track_name, artist_name)
        with self.cache_lock:
            self.analyzed_tracks.add(track_hash)

    def clear_cache(self):
        """Clear the analysis cache."""
        with self.cache_lock:
            self.analyzed_tracks.clear()
        logger.info("Analysis cache cleared")
    
    def search_and_download(self, track_name: str, artist_name: str) -> Optional[str]:
        """
        Search for a track on YouTube and download it as MP3 with enhanced error handling and quality validation.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            
        Returns:
            Path to downloaded MP3 file or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Create a unique directory for this analysis
                download_dir = self.create_unique_download_dir()
                
                # Create search query with better formatting
                search_query = f"{track_name} {artist_name} official audio"
                
                # Enhanced yt-dlp options with better quality settings
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
                    'quiet': False,  # Enable output for debugging
                    'no_warnings': False,  # Show warnings
                    'extract_flat': False,
                    'nocheckcertificate': True,  # Avoid SSL issues
                    'ignoreerrors': False,
                    'no_color': True,  # Avoid color codes in output
                    'extractor_args': {
                        'youtube': {
                            'skip': ['dash', 'live'],  # Skip live streams and DASH
                        }
                    }
                }
                
                logger.info(f"Searching for: {search_query} (attempt {attempt + 1}/{self.max_retries})")
                logger.info(f"Download directory: {download_dir}")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Search for the video with better error handling
                    try:
                        search_results = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                    except Exception as search_error:
                        logger.warning(f"Search failed, trying alternative query: {search_error}")
                        # Try alternative search query
                        alt_query = f"{track_name} {artist_name}"
                        search_results = ydl.extract_info(f"ytsearch1:{alt_query}", download=False)
                    
                    if not search_results or not search_results.get('entries'):
                        logger.warning(f"No results found for: {search_query}")
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return None
                    
                    # Get the first result and validate it
                    video_info = search_results['entries'][0]
                    video_url = video_info['url']
                    video_title = video_info.get('title', 'Unknown')
                    video_duration = video_info.get('duration', 0)
                    
                    # Validate video duration (skip very long or very short videos)
                    if video_duration and (video_duration < 30 or video_duration > 600):
                        logger.warning(f"Video duration unsuitable ({video_duration}s): {video_title}")
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        return None
                    
                    # Download the audio
                    logger.info(f"Downloading: {video_title} (duration: {video_duration}s)")
                    ydl.download([video_url])
                    
                    # Wait for audio conversion to complete
                    time.sleep(3)
                    
                    # Enhanced file detection - look for any audio file
                    audio_extensions = ['.mp3', '.m4a', '.wav', '.flac', '.ogg']
                    audio_files = []
                    
                    for ext in audio_extensions:
                        audio_files.extend([f for f in os.listdir(download_dir) if f.lower().endswith(ext)])
                    
                    if audio_files:
                        # Use the first audio file found
                        file_path = os.path.join(download_dir, audio_files[0])
                        
                        # Validate file size (should be reasonable for audio)
                        file_size = os.path.getsize(file_path)
                        if file_size < 10000:  # Less than 10KB is suspicious
                            logger.warning(f"Downloaded file too small ({file_size} bytes): {file_path}")
                            if attempt < self.max_retries - 1:
                                time.sleep(2 ** attempt)
                                continue
                            return None
                        
                        logger.info(f"Successfully downloaded: {file_path} ({file_size} bytes)")
                        return file_path
                    else:
                        logger.error(f"No audio files found in directory: {download_dir}")
                        # List what files are actually there for debugging
                        all_files = os.listdir(download_dir)
                        logger.error(f"Files in directory: {all_files}")
                        
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return None
                        
            except Exception as e:
                logger.error(f"Failed to download track (attempt {attempt + 1}/{self.max_retries}): {e}")
                import traceback
                logger.error(f"Download traceback: {traceback.format_exc()}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
        
        return None
    
    def validate_analysis_results(self, features: Dict[str, Any]) -> bool:
        """
        Validate that analysis results are within reasonable ranges and internally consistent.
        
        Args:
            features: Dictionary of extracted audio features
            
        Returns:
            True if features are valid, False otherwise
        """
        try:
            # Check that all required features are present
            required_features = [
                'tempo', 'beat_strength', 'rhythmic_stability', 'regularity',
                'valence', 'key', 'mode', 'energy', 'loudness', 'instrumentalness',
                'acousticness', 'speechiness', 'danceability'
            ]
            
            for feature in required_features:
                if feature not in features:
                    logger.error(f"Missing required feature: {feature}")
                    return False
            
            # Validate numeric ranges
            validations = [
                ('tempo', 40, 200),
                ('beat_strength', 0, 1),
                ('rhythmic_stability', 0, 1),
                ('regularity', 0, 1),
                ('valence', 0, 1),
                ('energy', 0, 1),
                ('loudness', -60, 0),
                ('instrumentalness', 0, 1),
                ('acousticness', 0, 1),
                ('speechiness', 0, 1),
                ('danceability', 0, 1)
            ]
            
            for feature, min_val, max_val in validations:
                value = features[feature]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    logger.error(f"Feature {feature} ({value}) outside valid range [{min_val}, {max_val}]")
                    return False
            
            # Validate categorical features
            valid_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'Unknown']
            valid_modes = ['major', 'minor', 'Unknown']
            
            if features['key'] not in valid_keys:
                logger.error(f"Invalid key: {features['key']}")
                return False
            
            if features['mode'] not in valid_modes:
                logger.error(f"Invalid mode: {features['mode']}")
                return False
            
            # Check for internal consistency
            # High instrumentalness should correlate with low speechiness
            if features['instrumentalness'] > 0.8 and features['speechiness'] > 0.7:
                logger.warning(f"Inconsistent features: high instrumentalness ({features['instrumentalness']}) with high speechiness ({features['speechiness']})")
            
            # High acousticness should correlate with lower energy (generally)
            if features['acousticness'] > 0.8 and features['energy'] > 0.9:
                logger.warning(f"Inconsistent features: high acousticness ({features['acousticness']}) with very high energy ({features['energy']})")
            
            # Tempo should correlate with danceability (to some extent)
            tempo_factor = features['tempo'] / 120.0
            if tempo_factor > 1.5 and features['danceability'] < 0.3:
                logger.warning(f"Low danceability ({features['danceability']}) for high tempo ({features['tempo']})")
            
            # All validations passed
            logger.info("Analysis results validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during feature validation: {e}")
            self.stats['validation_failures'] += 1
            return False
    
    def analyze_audio_features(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze audio file and extract features using librosa with enhanced preprocessing and feature extraction.
        
        Args:
            audio_file_path: Path to the MP3 file
            
        Returns:
            Dictionary with audio features or None if failed
        """
        try:
            logger.info(f"Analyzing audio file: {audio_file_path}")
            
            # Load the audio file with standard sample rate for consistency
            y, sr = librosa.load(audio_file_path, sr=22050)  # Standard sample rate
            
            # Validate audio quality and length
            duration = len(y) / sr
            if duration < 1.0:  # Skip very short tracks
                logger.warning(f"Audio too short ({duration:.2f}s), skipping analysis")
                return None
            if duration > 600:  # Limit to 10 minutes for performance
                logger.info(f"Audio very long ({duration:.2f}s), analyzing first 5 minutes")
                y = y[:int(5 * 60 * sr)]
            
            # Preprocessing: normalize audio and remove silence
            y = librosa.util.normalize(y)
            
            # Remove leading/trailing silence
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Check if audio is too quiet after trimming
            if np.max(np.abs(y)) < 0.01:
                logger.warning("Audio too quiet after trimming, skipping analysis")
                return None
            
            # Extract tempo and beat frames with enhanced parameters
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, 
                hop_length=512,
                start_bpm=120.0,
                tightness=100
            )
            
            # Validate tempo detection
            if tempo < 40 or tempo > 200:
                logger.warning(f"Unusual tempo detected: {tempo}, using fallback")
                tempo = 120.0  # Default tempo
            
            # Extract beat strength (RMS energy at beat frames)
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            if len(beat_frames) > 0 and max(beat_frames) < len(rms):
                beat_strength = np.mean(rms[beat_frames])
            else:
                beat_strength = np.mean(rms)
            
            # Extract rhythmic stability (standard deviation of beat intervals)
            if len(beat_frames) > 1:
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
                beat_intervals = np.diff(beat_times)
                rhythmic_stability = 1.0 / (1.0 + np.std(beat_intervals))
            else:
                rhythmic_stability = 0.5  # Default value
            
            # Extract regularity (autocorrelation of beat intervals)
            if len(beat_intervals) > 1:
                autocorr = np.correlate(beat_intervals, beat_intervals, mode='full')
                regularity = np.max(autocorr[len(autocorr)//2:]) / len(beat_intervals)
            else:
                regularity = 0.0
            
            # Enhanced harmonic/percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
            
            # Spectral contrast (brightness vs darkness)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
            spectral_contrast_mean = np.mean(spectral_contrast)
            
            # Harmonic content ratio (more harmonic = more positive)
            harmonic_energy = np.sum(y_harmonic**2)
            percussive_energy = np.sum(y_percussive**2)
            total_energy = harmonic_energy + percussive_energy + 1e-6
            harmonic_ratio = harmonic_energy / total_energy
            
            # Enhanced valence calculation using multiple features
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # Spectral rolloff (high frequency content)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            
            # MFCC features for timbral characteristics
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # --- Improved Valence Calculation ---
            # Penalize for minor key, low energy, and high dynamic range
            rms_var = np.var(rms)
            is_minor = (mode == 'minor')
            valence = float(np.clip(
                (harmonic_ratio * 0.2 +
                 spectral_contrast_mean * 0.15 +
                 (spectral_centroid_mean / sr) * 0.15 +
                 (spectral_rolloff_mean / sr) * 0.1 +
                 (0.2 if not is_minor else 0.0) +
                 (0.2 if energy > 0.3 else 0.0) -
                 (0.2 if rms_var > 0.02 else 0.0)), 0.0, 1.0
            ))
            
            # Enhanced key and mode detection
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
            
            # Use librosa's key detection with fallback
            try:
                # For newer versions of librosa
                key_raw, mode_raw = librosa.feature.key_mode(chroma)
            except AttributeError:
                # Fallback for older versions
                try:
                    # Alternative approach for key detection
                    chroma_sum = np.sum(chroma, axis=1)
                    key_raw = np.argmax(chroma_sum)
                    # Simple mode detection based on minor third
                    mode_raw = 0  # Default to major
                except:
                    key_raw = 0
                    mode_raw = 0
            
            # Convert key and mode to strings
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            mode_names = ['major', 'minor']
            
            key = key_names[key_raw] if key_raw < len(key_names) else 'Unknown'
            mode = mode_names[mode_raw] if mode_raw < len(mode_names) else 'Unknown'
            
            # Enhanced energy calculation
            energy = float(np.mean(rms))

            # Enhanced loudness calculation (perceptual loudness)
            loudness = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))

            # Enhanced instrumentalness calculation
            spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
            spectral_flatness_mean = np.mean(spectral_flatness)
            
            # Harmonic/percussive ratio for instrumentalness
            harmonic_percussive_ratio = harmonic_energy / (percussive_energy + 1e-6)
            
            # Spectral bandwidth (instrumental music tends to have wider bandwidth)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            
            # --- Improved Instrumentalness ---
            # Use vocal activity detection: penalize if vocals are detected
            # Use bandpass filter for vocal range (300–3400 Hz)
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            vocal_band = (freqs > 300) & (freqs < 3400)
            vocal_energy = np.mean(S[vocal_band, :])
            total_energy_stft = np.mean(S)
            vocal_activity_ratio = vocal_energy / (total_energy_stft + 1e-6)
            # If vocal activity is high, lower instrumentalness
            instrumentalness = float(np.clip(
                (harmonic_percussive_ratio * 0.4 +
                 spectral_flatness_mean * 0.2 +
                 (spectral_bandwidth_mean / sr) * 0.1 +
                 (0.3 if vocal_activity_ratio < 0.15 else 0.0)), 0.0, 1.0
            ))

            # Enhanced acousticness calculation
            # Acoustic instruments typically have more harmonic content and less spectral flatness
            acousticness = float(np.clip(
                (harmonic_ratio * 0.6 + 
                 (1.0 - spectral_flatness_mean) * 0.4), 0.0, 1.0
            ))

            # --- Improved Speechiness ---
            # Use pitch tracking: if melodic, lower speechiness
            try:
                pitches, voiced_flags, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                voiced_ratio = np.mean(voiced_flags.astype(float))
                pitch_var = np.nanvar(pitches)
            except Exception:
                voiced_ratio = 0.0
                pitch_var = 0.0
            # If voiced_ratio is high and pitch_var is high, it's likely singing, not speech
            speechiness = float(np.clip(
                (spectral_centroid_var * 0.2 +
                 zcr * 0.2 +
                 spectral_rolloff_var * 0.2 +
                 (0.4 if (voiced_ratio < 0.2 and pitch_var < 100) else 0.0)), 0.0, 1.0
            ))

            # --- Improved Danceability ---
            # Penalize for high dynamic range, irregular beat, and long sections without strong onsets
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_env_var = np.var(onset_env)
            strong_onset_ratio = np.mean(onset_env > np.percentile(onset_env, 75))
            # Windowed beat regularity
            if len(beat_frames) > 1:
                beat_intervals = np.diff(librosa.frames_to_time(beat_frames, sr=sr, hop_length=512))
                beat_regularity = 1.0 / (1.0 + np.std(beat_intervals))
            else:
                beat_regularity = 0.5
            danceability = float(np.clip(
                (tempo_factor * 0.2 +
                 beat_regularity * 0.2 +
                 rhythm_strength * 0.2 +
                 strong_onset_ratio * 0.2 -
                 (0.2 if rms_var > 0.02 else 0.0)), 0.0, 1.0
            ))

            # Validate and clamp all features to reasonable ranges
            features = {
                'tempo': round(float(np.clip(tempo_scalar, 40, 200)), 2),
                'beat_strength': round(float(np.clip(beat_strength, 0, 1)), 2),
                'rhythmic_stability': round(float(np.clip(rhythmic_stability, 0, 1)), 2),
                'regularity': round(float(np.clip(regularity, 0, 1)), 2),
                'valence': round(float(np.clip(valence, 0, 1)), 2),
                'key': key,
                'mode': mode,
                'energy': round(float(np.clip(energy, 0, 1)), 2),
                'loudness': round(float(np.clip(loudness, -60, 0)), 2),
                'instrumentalness': round(float(np.clip(instrumentalness, 0, 1)), 2),
                'acousticness': round(float(np.clip(acousticness, 0, 1)), 2),
                'speechiness': round(float(np.clip(speechiness, 0, 1)), 2),
                'danceability': round(float(np.clip(danceability, 0, 1)), 2)
            }
            
            logger.info(f"Audio analysis completed: {features}")
            
            # Console output for debugging
            print(f"\n🎵 AUDIO ANALYSIS RESULTS:")
            for k, v in features.items():
                print(f"   {k}: {v}")
            print(f"   File: {audio_file_path}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Sample Rate: {sr}Hz\n")
            
            # Validate the analysis results
            if self.validate_analysis_results(features):
                return features
            else:
                logger.error("Analysis results failed validation")
                self.stats['validation_failures'] += 1
                return None
            
        except Exception as e:
            logger.error(f"Failed to analyze audio: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def cleanup_audio_file(self, audio_file_path: str):
        """Remove the downloaded audio file and its directory."""
        try:
            if os.path.exists(audio_file_path):
                # Remove the file
                os.remove(audio_file_path)
                logger.info(f"Cleaned up file: {audio_file_path}")
                
                # Remove the directory if it's empty
                directory = os.path.dirname(audio_file_path)
                if os.path.exists(directory) and not os.listdir(directory):
                    os.rmdir(directory)
                    logger.info(f"Cleaned up directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to cleanup audio file: {e}")
    
    def analyze_track_with_retries(self, track_name: str, artist_name: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a track with retry mechanism.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            
        Returns:
            Dictionary with audio features or None if failed
        """
        audio_file_path = None
        for attempt in range(self.max_retries):
            try:
                # Download the track
                audio_file_path = self.search_and_download(track_name, artist_name)
                if not audio_file_path:
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying analysis for {track_name} by {artist_name} (attempt {attempt + 2})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    # Track download failure
                    self.stats['download_failures'] += 1
                    return None
                
                # Analyze the audio
                features = self.analyze_audio_features(audio_file_path)
                print(f"[DEBUG] analyze_track_with_retries: features = {features is not None}")
                if features:
                    print(f"[DEBUG] analyze_track_with_retries: features keys = {list(features.keys())}")
                    # Mark as analyzed in cache
                    self.mark_track_analyzed(track_name, artist_name)
                    return features
                else:
                    print(f"[DEBUG] analyze_track_with_retries: no features returned")
                    if attempt < self.max_retries - 1:
                        logger.info(f"Retrying analysis for {track_name} by {artist_name} (attempt {attempt + 2})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    # Track analysis failure
                    self.stats['analysis_failures'] += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Analysis failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                # Track analysis failure
                self.stats['analysis_failures'] += 1
                return None
            finally:
                # Clean up the downloaded file
                if audio_file_path:
                    self.cleanup_audio_file(audio_file_path)
        
        return None
    
    def submit_analysis_task(self, track_name: str, artist_name: str, track_id: str, status_callback=None):
        """
        Submit an analysis task to the thread pool.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            track_id: Spotify track ID for status updates
            status_callback: Optional callback function to update analysis status
            
        Returns:
            Future object for the analysis task
        """
        # Update statistics
        self.stats['total_analyses'] += 1
        
        # Check cache first
        if self.is_track_analyzed(track_name, artist_name):
            logger.info(f"Track {track_name} by {artist_name} already analyzed, skipping")
            self.stats['cached_hits'] += 1
            if status_callback:
                status_callback(track_id, 'completed')
            return None
        
        # Submit to thread pool
        future = self.executor.submit(self._analyze_track_task, track_name, artist_name, track_id, status_callback)
        logger.info(f"Submitted analysis task for {track_name} by {artist_name}")
        return future
    
    def _analyze_track_task(self, track_name: str, artist_name: str, track_id: str, status_callback=None):
        """
        Internal method to run analysis task with status updates.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            track_id: Spotify track ID for status updates
            status_callback: Optional callback function to update analysis status
        """
        start_time = time.time()
        try:
            # Update status to in_progress
            if status_callback:
                status_callback(track_id, 'in_progress')
            
            # Perform analysis with retries (this handles its own file cleanup)
            features = self.analyze_track_with_retries(track_name, artist_name)
            print(f"[DEBUG] _analyze_track_task: features = {features is not None}")
            
            if features:
                print(f"[DEBUG] _analyze_track_task: features keys = {list(features.keys())}")
                # Update statistics
                self.stats['successful_analyses'] += 1
                duration = time.time() - start_time
                self.stats['total_duration'] += duration
                self.stats['average_duration'] = self.stats['total_duration'] / self.stats['successful_analyses']
                
                # Update status to completed
                if status_callback:
                    status_callback(track_id, 'completed')
                logger.info(f"Analysis completed successfully for {track_name} by {artist_name} in {duration:.2f}s")
                return features
            else:
                print(f"[DEBUG] _analyze_track_task: no features returned")
                # Update statistics
                self.stats['failed_analyses'] += 1
                self.stats['analysis_failures'] += 1
                
                # Update status to failed
                if status_callback:
                    status_callback(track_id, 'failed')
                logger.error(f"Analysis failed for {track_name} by {artist_name}")
                return None
                
        except Exception as e:
            # Update statistics
            self.stats['failed_analyses'] += 1
            self.stats['analysis_failures'] += 1
            
            # Update status to failed
            if status_callback:
                status_callback(track_id, 'failed')
            logger.error(f"Analysis task failed for {track_name} by {artist_name}: {e}")
            return None
    
    def get_active_tasks_count(self) -> int:
        """Get the number of active analysis tasks."""
        return len([f for f in self.executor._threads if f.is_alive()])
    
    def print_analysis_stats(self):
        """Print current analysis statistics in a formatted way."""
        stats = self.get_analysis_stats()
        print("\n📊 AUDIO ANALYSIS STATISTICS:")
        print(f"   Total Analyses: {stats['total_analyses']}")
        print(f"   Successful: {stats['successful_analyses']}")
        print(f"   Failed: {stats['failed_analyses']}")
        print(f"   Cache Hits: {stats['cached_hits']}")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        print(f"   Download Failures: {stats['download_failures']}")
        print(f"   Analysis Failures: {stats['analysis_failures']}")
        print(f"   Validation Failures: {stats['validation_failures']}")
        print(f"   Average Duration: {stats['average_duration']:.2f}s")
        print(f"   Total Duration: {stats['total_duration']:.2f}s")
        print(f"   Active Tasks: {self.get_active_tasks_count()}")
        print()

    def wait_for_completion(self, timeout: int = 300):
        """Wait for all pending tasks to complete."""
        logger.info(f"Waiting for {self.get_active_tasks_count()} tasks to complete...")
        self.executor.shutdown(wait=True, timeout=timeout)
        logger.info("All analysis tasks completed")

if __name__ == "__main__":
    # Test the audio analyzer
    analyzer = AudioAnalyzer()
    
    # Test with a sample track
    test_track = "Bohemian Rhapsody"
    test_artist = "Queen"
    
    print(f"Testing audio analysis for: {test_track} by {test_artist}")
    features = analyzer.analyze_track_with_retries(test_track, test_artist)
    
    if features:
        print("✅ Audio analysis successful!")
        print("Features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        # Print statistics
        analyzer.print_analysis_stats()
    else:
        print("❌ Audio analysis failed")
        analyzer.print_analysis_stats() 
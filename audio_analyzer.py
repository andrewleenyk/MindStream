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
        """Initialize the audio analyzer with thread pooling and retry settings."""
        self.base_download_dir = base_download_dir
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.analyzed_cache = set()  # Track analyzed tracks
        self.cache_lock = Lock()  # Thread-safe cache access
        self.ensure_base_download_dir()
        
        # Thread pool for concurrent analysis
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
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
    
    def is_track_analyzed(self, track_name: str, artist_name: str) -> bool:
        """Check if a track has already been analyzed (thread-safe)."""
        track_hash = self.get_track_hash(track_name, artist_name)
        with self.cache_lock:
            return track_hash in self.analyzed_cache
    
    def mark_track_analyzed(self, track_name: str, artist_name: str):
        """Mark a track as analyzed (thread-safe)."""
        track_hash = self.get_track_hash(track_name, artist_name)
        with self.cache_lock:
            self.analyzed_cache.add(track_hash)
    
    def clear_cache(self):
        """Clear the analysis cache."""
        with self.cache_lock:
            self.analyzed_cache.clear()
        logger.info("Analysis cache cleared")
    
    def search_and_download(self, track_name: str, artist_name: str) -> Optional[str]:
        """
        Search for a track on YouTube and download it as MP3 with retries.
        
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
                
                # Create search query
                search_query = f"{track_name} {artist_name} audio"
                
                # Configure yt-dlp options
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
                    'quiet': False,  # Enable output for debugging
                    'no_warnings': False,  # Show warnings
                    'extract_flat': False,
                }
                
                logger.info(f"Searching for: {search_query} (attempt {attempt + 1}/{self.max_retries})")
                logger.info(f"Download directory: {download_dir}")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Search for the video
                    search_results = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                    
                    if not search_results.get('entries'):
                        logger.warning(f"No results found for: {search_query}")
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return None
                    
                    # Get the first result
                    video_info = search_results['entries'][0]
                    video_url = video_info['url']
                    
                    # Download the audio
                    logger.info(f"Downloading: {video_info.get('title', 'Unknown')}")
                    ydl.download([video_url])
                    
                    # Small delay to ensure audio conversion is complete
                    time.sleep(2)
                    
                    # Find the downloaded file - look for any .mp3 file in the directory
                    mp3_files = [f for f in os.listdir(download_dir) if f.endswith('.mp3')]
                    
                    if mp3_files:
                        # Use the first .mp3 file found
                        file_path = os.path.join(download_dir, mp3_files[0])
                        logger.info(f"Successfully downloaded: {file_path}")
                        return file_path
                    else:
                        logger.error(f"No MP3 files found in directory: {download_dir}")
                        # List what files are actually there for debugging
                        all_files = os.listdir(download_dir)
                        logger.error(f"Files in directory: {all_files}")
                        
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        return None
                        
            except Exception as e:
                logger.error(f"Failed to download track (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None
        
        return None
    
    def analyze_audio_features(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze audio file and extract features using librosa.
        
        Args:
            audio_file_path: Path to the MP3 file
            
        Returns:
            Dictionary with audio features or None if failed
        """
        try:
            logger.info(f"Analyzing audio file: {audio_file_path}")
            
            # Load the audio file
            y, sr = librosa.load(audio_file_path, sr=None)
            
            # Extract tempo and beat frames
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            
            # Extract beat strength (RMS energy at beat frames)
            beat_strength = np.mean(librosa.feature.rms(y=y)[0, beat_frames])
            
            # Extract rhythmic stability (standard deviation of beat intervals)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            beat_intervals = np.diff(beat_times)
            rhythmic_stability = 1.0 / (1.0 + np.std(beat_intervals))
            
            # Extract regularity (autocorrelation of beat intervals)
            if len(beat_intervals) > 1:
                autocorr = np.correlate(beat_intervals, beat_intervals, mode='full')
                regularity = np.max(autocorr[len(autocorr)//2:]) / len(beat_intervals)
            else:
                regularity = 0.0
            
            # Extract valence (positive emotion) using spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Simple valence approximation based on spectral features
            valence = np.mean(spectral_centroids) / (np.mean(spectral_rolloff) + 1e-6)
            valence = np.clip(valence, 0.0, 1.0)
            
            # Extract key and mode
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Use librosa's key detection (fixed method)
            try:
                # For newer versions of librosa
                key_raw, mode_raw = librosa.feature.key_mode(chroma)
            except AttributeError:
                # Fallback for older versions or different method
                try:
                    # Alternative approach for key detection
                    key_raw = np.argmax(np.sum(chroma, axis=1))
                    mode_raw = 0  # Default to major
                except:
                    key_raw = 0
                    mode_raw = 0
            
            # Convert key and mode to strings
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            mode_names = ['major', 'minor']
            
            key = key_names[key_raw] if key_raw < len(key_names) else 'Unknown'
            mode = mode_names[mode_raw] if mode_raw < len(mode_names) else 'Unknown'
            
            # Compute energy (mean RMS)
            rms = librosa.feature.rms(y=y)[0]
            energy = float(np.mean(rms))

            # Compute loudness (in dB)
            loudness = float(np.mean(librosa.amplitude_to_db(rms, ref=np.max)))

            # Instrumentalness (heuristic: low MFCC variation and low zero-crossing rate)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.mean(np.var(mfcc, axis=1))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            instrumentalness = float(np.clip(1.0 - (mfcc_var + zcr), 0.0, 1.0))

            # Acousticness (heuristic: low spectral centroid and rolloff)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            acousticness = float(np.clip(1.0 - (np.mean(spectral_centroids) / (np.mean(spectral_rolloff) + 1e-6)), 0.0, 1.0))

            # Speechiness (heuristic: high MFCC variation and high zero-crossing rate)
            speechiness = float(np.clip((mfcc_var + zcr) / 2.0, 0.0, 1.0))

            # Danceability (heuristic: combine tempo, beat strength, and regularity)
            danceability = float(np.clip((tempo / 200.0 + beat_strength + regularity) / 3.0, 0.0, 1.0))

            # Compile features, rounded to 2 decimal places
            features = {
                'tempo': round(float(tempo), 2),
                'beat_strength': round(float(beat_strength), 2),
                'rhythmic_stability': round(float(rhythmic_stability), 2),
                'regularity': round(float(regularity), 2),
                'valence': round(float(valence), 2),
                'key': key,
                'mode': mode,
                'energy': round(energy, 2),
                'loudness': round(loudness, 2),
                'instrumentalness': round(instrumentalness, 2),
                'acousticness': round(acousticness, 2),
                'speechiness': round(speechiness, 2),
                'danceability': round(danceability, 2)
            }
            
            logger.info(f"Audio analysis completed: {features}")
            
            # Console output for debugging
            print(f"\n🎵 AUDIO ANALYSIS RESULTS:")
            for k, v in features.items():
                print(f"   {k}: {v}")
            print(f"   File: {audio_file_path}\n")
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to analyze audio: {e}")
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
                    return None
                    
            except Exception as e:
                logger.error(f"Analysis failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
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
        # Check cache first
        if self.is_track_analyzed(track_name, artist_name):
            logger.info(f"Track {track_name} by {artist_name} already analyzed, skipping")
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
        try:
            # Update status to in_progress
            if status_callback:
                status_callback(track_id, 'in_progress')
            
            # Perform analysis with retries (this handles its own file cleanup)
            features = self.analyze_track_with_retries(track_name, artist_name)
            print(f"[DEBUG] _analyze_track_task: features = {features is not None}")
            
            if features:
                print(f"[DEBUG] _analyze_track_task: features keys = {list(features.keys())}")
                # Update status to completed
                if status_callback:
                    status_callback(track_id, 'completed')
                logger.info(f"Analysis completed successfully for {track_name} by {artist_name}")
                return features
            else:
                print(f"[DEBUG] _analyze_track_task: no features returned")
                # Update status to failed
                if status_callback:
                    status_callback(track_id, 'failed')
                logger.error(f"Analysis failed for {track_name} by {artist_name}")
                return None
                
        except Exception as e:
            # Update status to failed
            if status_callback:
                status_callback(track_id, 'failed')
            logger.error(f"Analysis task failed for {track_name} by {artist_name}: {e}")
            return None
    
    def get_active_tasks_count(self) -> int:
        """Get the number of active analysis tasks."""
        return len([f for f in self.executor._threads if f.is_alive()])
    
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
    else:
        print("❌ Audio analysis failed") 
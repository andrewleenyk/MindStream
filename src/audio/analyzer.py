"""
Audio Analyzer Module for Mindstream
Integrates Mozart enhanced audio analysis with Mindstream's database system.
"""

import os
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .mozart_analyzer import MozartAudioAnalyzer
from .downloader import AudioDownloader
from .validator import AudioValidator
from .statistics import AudioStatistics

# Try to import config, fall back to a simple config class if not available
try:
    from ..core.config import Config
except ImportError:
    # Fallback configuration class
    class Config:
        def get(self, section, key, default=None):
            if section == 'audio' and key == 'temp_directory':
                return 'temp_audio'
            elif section == 'audio' and key == 'max_retries':
                return 3
            elif section == 'audio' and key == 'max_duration':
                return 600
            elif section == 'audio' and key == 'min_duration':
                return 1.0
            elif section == 'audio' and key == 'min_sample_rate':
                return 8000
            elif section == 'audio' and key == 'max_sample_rate':
                return 48000
            return default

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Main audio analyzer that integrates Mozart enhanced analysis with Mindstream."""
    
    def __init__(self, config: Config = None):
        """
        Initialize the audio analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.mozart_analyzer = MozartAudioAnalyzer()
        self.downloader = AudioDownloader(self.config)
        self.validator = AudioValidator(self.config)
        self.statistics = AudioStatistics()
        
        # Analysis state
        self.is_analyzing = False
        self.current_track = None
        self.analysis_progress = 0
        self.total_tracks = 0
        self.processed_tracks = 0
        self.failed_tracks = 0
        
        logger.info("✅ Audio Analyzer initialized with Mozart enhanced analysis")
    
    def analyze_track(self, track_info: Dict[str, Any], audio_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single track using Mozart enhanced analysis.
        
        Args:
            track_info: Track metadata from Spotify
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting Mozart analysis for: {track_info.get('name', 'Unknown')}")
            
            # Validate audio file
            if not self.validator.is_valid_audio_file(audio_file_path):
                logger.error(f"Invalid audio file: {audio_file_path}")
                return None
            
            # Extract enhanced features using Mozart
            features = self.mozart_analyzer.extract_all_features(audio_file_path)
            if not features:
                logger.error(f"Failed to extract features from: {audio_file_path}")
                return None
            
            # Combine track info with analysis results
            analysis_result = {
                'track_id': track_info.get('id'),
                'name': track_info.get('name'),
                'artist': track_info.get('artists', [{}])[0].get('name') if track_info.get('artists') else 'Unknown',
                'album': track_info.get('album', {}).get('name') if track_info.get('album') else 'Unknown',
                'duration_ms': track_info.get('duration_ms'),
                'popularity': track_info.get('popularity'),
                'audio_file_path': audio_file_path,
                'analysis_timestamp': time.time(),
                **features
            }
            
            logger.info(f"Mozart analysis completed successfully for: {track_info.get('name', 'Unknown')}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing track {track_info.get('name', 'Unknown')}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def analyze_tracks_batch(self, tracks: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Analyze multiple tracks in parallel using Mozart enhanced analysis.
        
        Args:
            tracks: List of track metadata from Spotify
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of analysis results
        """
        results = []
        self.total_tracks = len(tracks)
        self.processed_tracks = 0
        self.failed_tracks = 0
        self.is_analyzing = True
        
        logger.info(f"Starting batch analysis of {len(tracks)} tracks with Mozart enhanced analysis")
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit analysis tasks
                future_to_track = {}
                for track in tracks:
                    track_id = track.get('id')
                    if not track_id:
                        continue
                    
                    # Download audio if needed
                    audio_file_path = self.downloader.get_audio_file_path(track_id)
                    if not os.path.exists(audio_file_path):
                        logger.info(f"Downloading audio for: {track.get('name', 'Unknown')}")
                        download_success = self.downloader.download_track(track)
                        if not download_success:
                            logger.error(f"Failed to download audio for: {track.get('name', 'Unknown')}")
                            self.failed_tracks += 1
                            continue
                    
                    # Submit analysis task
                    future = executor.submit(self.analyze_track, track, audio_file_path)
                    future_to_track[future] = track
                
                # Collect results
                for future in as_completed(future_to_track):
                    track = future_to_track[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            logger.info(f"Analysis completed: {track.get('name', 'Unknown')}")
                        else:
                            logger.error(f"Analysis failed: {track.get('name', 'Unknown')}")
                            self.failed_tracks += 1
                    except Exception as e:
                        logger.error(f"Exception in analysis for {track.get('name', 'Unknown')}: {e}")
                        self.failed_tracks += 1
                    
                    self.processed_tracks += 1
                    self.analysis_progress = (self.processed_tracks / self.total_tracks) * 100
                    
                    # Update current track for progress tracking
                    self.current_track = track.get('name', 'Unknown')
        
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            self.is_analyzing = False
            self.current_track = None
        
        # Generate analysis statistics
        stats = self.statistics.generate_analysis_stats(results)
        logger.info(f"Batch analysis completed. Processed: {len(results)}, Failed: {self.failed_tracks}")
        logger.info(f"Analysis statistics: {stats}")
        
        return results
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """
        Get current analysis status.
        
        Returns:
            Dictionary containing analysis status information
        """
        return {
            'is_analyzing': self.is_analyzing,
            'current_track': self.current_track,
            'progress': self.analysis_progress,
            'total_tracks': self.total_tracks,
            'processed_tracks': self.processed_tracks,
            'failed_tracks': self.failed_tracks
        }
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        try:
            temp_dir = self.config.get('audio', 'temp_directory', 'temp_audio')
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info("Temporary audio files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    def submit_analysis_task(self, track_name: str, artist_name: str, track_id: str, status_callback=None):
        """
        Submit an analysis task for compatibility with existing tracker code.
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist
            track_id: Spotify track ID
            status_callback: Optional callback function for status updates
            
        Returns:
            Future object for the analysis task
        """
        from concurrent.futures import ThreadPoolExecutor
        
        # Create a simple track info dict
        track_info = {
            'id': track_id,
            'name': track_name,
            'artists': [{'name': artist_name}],
            'album': {'name': 'Unknown'},
            'duration_ms': 0,
            'popularity': 0
        }
        
        # Create thread pool for this task
        executor = ThreadPoolExecutor(max_workers=1)
        
        def analysis_task():
            try:
                if status_callback:
                    status_callback(track_id, "downloading")
                
                # Download the track
                download_success = self.downloader.download_track(track_info)
                if not download_success:
                    if status_callback:
                        status_callback(track_id, "download_failed")
                    return None
                
                if status_callback:
                    status_callback(track_id, "analyzing")
                
                # Get the audio file path
                audio_file_path = self.downloader.get_audio_file_path(track_id)
                
                # Analyze the track
                result = self.analyze_track(track_info, audio_file_path)
                
                if result:
                    if status_callback:
                        status_callback(track_id, "completed")
                    # Return just the features for compatibility
                    return {k: v for k, v in result.items() if k not in ['track_id', 'name', 'artist', 'album', 'duration_ms', 'popularity', 'audio_file_path', 'analysis_timestamp']}
                else:
                    if status_callback:
                        status_callback(track_id, "failed")
                    return None
                    
            except Exception as e:
                logger.error(f"Error in analysis task for {track_name}: {e}")
                if status_callback:
                    status_callback(track_id, "failed")
                return None
        
        # Submit the task
        future = executor.submit(analysis_task)
        return future
    
    def get_active_tasks_count(self) -> int:
        """Get the number of active analysis tasks (for compatibility)."""
        return 1 if self.is_analyzing else 0
    
    def wait_for_completion(self, timeout: int = 300):
        """Wait for analysis completion (for compatibility)."""
        start_time = time.time()
        while self.is_analyzing and (time.time() - start_time) < timeout:
            time.sleep(1) 
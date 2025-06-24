"""
Audio Downloader Module for Mindstream
Downloads audio files for analysis using yt-dlp.
"""

import os
import logging
import hashlib
from typing import Dict, Any, Optional
import yt_dlp

logger = logging.getLogger(__name__)


class AudioDownloader:
    """Downloads audio files for analysis."""
    
    def __init__(self, config):
        """
        Initialize the audio downloader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.base_download_dir = config.get('audio', 'temp_directory', 'temp_audio')
        self.max_retries = config.get('audio', 'max_retries', 3)
        self.max_duration = config.get('audio', 'max_duration', 600)  # 10 minutes
        
        # Ensure download directory exists
        os.makedirs(self.base_download_dir, exist_ok=True)
        
        logger.info(f"✅ Audio Downloader initialized with directory: {self.base_download_dir}")
    
    def get_audio_file_path(self, track_id: str) -> str:
        """
        Get the expected audio file path for a track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Path where the audio file should be stored
        """
        filename = f"{track_id}.mp3"
        return os.path.join(self.base_download_dir, filename)
    
    def download_track(self, track_info: Dict[str, Any]) -> bool:
        """
        Download audio for a track using yt-dlp.
        
        Args:
            track_info: Track metadata from Spotify
            
        Returns:
            True if download successful, False otherwise
        """
        track_id = track_info.get('id')
        track_name = track_info.get('name', 'Unknown')
        artist_name = track_info.get('artists', [{}])[0].get('name', 'Unknown') if track_info.get('artists') else 'Unknown'
        
        if not track_id:
            logger.error("No track ID provided")
            return False
        
        # Check if file already exists
        audio_file_path = self.get_audio_file_path(track_id)
        logger.info(f"Expected output file: {audio_file_path}")
        if os.path.exists(audio_file_path):
            logger.info(f"Audio file already exists: {audio_file_path}")
            return True
        
        # Create search query (simpler: just track name and artist)
        search_query = f"{track_name} {artist_name}"
        
        logger.info(f"Downloading audio for: {track_name} by {artist_name}")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_file_path.replace('.mp3', '.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': False,  # Show yt-dlp output for debugging
            'no_warnings': False,  # Show warnings for debugging
            'extract_flat': False,
            'max_duration': self.max_duration,
            'retries': self.max_retries,
            'fragment_retries': self.max_retries,
            'skip_unavailable_fragments': True,
            'ignoreerrors': False,
            'nocheckcertificate': True,
            'prefer_ffmpeg': True,
            'geo_bypass': True,
            'nocheckcertificate': True,
            'prefer_ffmpeg': True,
            'default_search': 'ytsearch',
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search and download
                search_results = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                
                if not search_results or 'entries' not in search_results or not search_results['entries']:
                    logger.error(f"No search results found for: {search_query}")
                    return False
                
                # Get the first result
                video_info = search_results['entries'][0]
                if not video_info:
                    logger.error(f"No video info found for: {search_query}")
                    return False
                
                # Download the video
                ydl.download([video_info['webpage_url']])
                
                # Debug: List all files in the download directory
                logger.info(f"Listing files in {self.base_download_dir} after download:")
                for fname in os.listdir(self.base_download_dir):
                    fpath = os.path.join(self.base_download_dir, fname)
                    logger.info(f" - {fname} ({os.path.getsize(fpath)} bytes)")
                
                # Verify the file was downloaded
                if os.path.exists(audio_file_path):
                    file_size = os.path.getsize(audio_file_path)
                    if file_size > 1024:  # At least 1KB
                        logger.info(f"Successfully downloaded: {audio_file_path} ({file_size} bytes)")
                        return True
                    else:
                        logger.error(f"Downloaded file too small: {file_size} bytes")
                        return False
                else:
                    logger.error(f"Download completed but file not found: {audio_file_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error downloading audio for {track_name} by {artist_name}: {e}")
            # Extra debug: list files even on error
            try:
                logger.info(f"Listing files in {self.base_download_dir} after error:")
                for fname in os.listdir(self.base_download_dir):
                    fpath = os.path.join(self.base_download_dir, fname)
                    logger.info(f" - {fname} ({os.path.getsize(fpath)} bytes)")
            except Exception as e2:
                logger.error(f"Error listing files after download error: {e2}")
            return False
    
    def cleanup_audio_file(self, track_id: str) -> bool:
        """
        Clean up audio file for a track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            audio_file_path = self.get_audio_file_path(track_id)
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                logger.info(f"Cleaned up audio file: {audio_file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up audio file for {track_id}: {e}")
            return False
    
    def cleanup_all_files(self) -> int:
        """
        Clean up all downloaded audio files.
        
        Returns:
            Number of files cleaned up
        """
        try:
            cleaned_count = 0
            if os.path.exists(self.base_download_dir):
                for filename in os.listdir(self.base_download_dir):
                    file_path = os.path.join(self.base_download_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} audio files")
            return cleaned_count
        except Exception as e:
            logger.error(f"Error cleaning up audio files: {e}")
            return 0
    
    def get_download_status(self, track_id: str) -> Dict[str, Any]:
        """
        Get download status for a track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dictionary with download status information
        """
        audio_file_path = self.get_audio_file_path(track_id)
        
        status = {
            'track_id': track_id,
            'file_path': audio_file_path,
            'exists': os.path.exists(audio_file_path),
            'file_size': 0
        }
        
        if status['exists']:
            status['file_size'] = os.path.getsize(audio_file_path)
        
        return status 
#!/usr/bin/env python3
"""
Spotify Continuous Tracker

A single Python script that:
- Automatically refreshes access_token every 55 minutes
- Polls /me/player/currently-playing every 5 seconds
- Uses latest valid token for all API calls
- Logs track info to console
- Downloads and analyzes audio features for new tracks
"""

import time
import threading
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any

# Import our existing modules
from auth import token_manager, get_spotify_headers
from tracker.spotify_api import SpotifyAPIClient
from supabase_database import SupabaseDatabase
from audio_analyzer import AudioAnalyzer

class SpotifyTracker:
    """Continuous Spotify tracking with automatic token refresh and enhanced audio analysis."""
    
    def __init__(self):
        self.client = SpotifyAPIClient()
        self.database = SupabaseDatabase()
        self.audio_analyzer = AudioAnalyzer(max_concurrent=2, max_retries=3)
        self.running = False
        self.last_track_id = None
        self.analysis_futures = {}  # Track analysis futures
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        # Wait for analysis tasks to complete
        self.audio_analyzer.wait_for_completion(timeout=30)
    
    def _token_refresh_worker(self):
        """Background worker to refresh tokens every 55 minutes."""
        print("🔄 Token refresh worker started (55-minute interval)")
        
        while self.running:
            try:
                # Get a fresh token (this will refresh if needed)
                token = token_manager.get_access_token()
                print(f"✅ Token refreshed at {datetime.now().strftime('%H:%M:%S')}")
                
                # Sleep for 55 minutes
                time.sleep(55 * 60)
                
            except Exception as e:
                print(f"❌ Token refresh error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _format_track_info(self, track_data: Dict[str, Any]) -> str:
        """Format track information for console output."""
        if not track_data or 'item' not in track_data:
            return "Nothing playing"
        
        item = track_data['item']
        track_name = item.get('name', 'Unknown Track')
        artists = [artist.get('name', 'Unknown') for artist in item.get('artists', [])]
        artist_names = ', '.join(artists)
        is_playing = track_data.get('is_playing', False)
        
        # Get progress info
        progress_ms = track_data.get('progress_ms', 0)
        duration_ms = item.get('duration_ms', 0)
        
        if duration_ms > 0:
            progress_sec = progress_ms // 1000
            duration_sec = duration_ms // 1000
            progress_str = f" ({progress_sec}s/{duration_sec}s)"
        else:
            progress_str = ""
        
        status = "▶️" if is_playing else "⏸️"
        
        return f"{status} {track_name} by {artist_names}{progress_str}"
    
    def _log_track_change(self, track_data: Optional[Dict[str, Any]]):
        """Log track information when it changes."""
        current_track_id = None
        
        if track_data and 'item' in track_data:
            current_track_id = track_data['item'].get('id')
        
        # Only log if track changed or if nothing is playing
        if current_track_id != self.last_track_id:
            timestamp = datetime.now().strftime('%H:%M:%S')
            track_info = self._format_track_info(track_data)
            print(f"[{timestamp}] {track_info}")
            
            # Get comprehensive track data
            comprehensive_data = self.client.get_comprehensive_track_data(track_data)
            
            if comprehensive_data:
                print(f"[{timestamp}] 📊 Track Data:")
                # Show key data points
                data_summary = {
                    'timestamp': comprehensive_data.get('timestamp'),
                    'track_id': comprehensive_data.get('track_id'),
                    'track_name': comprehensive_data.get('track_name'),
                    'primary_artist': comprehensive_data.get('primary_artist'),
                    'is_playing': comprehensive_data.get('is_playing'),
                    'progress_ms': comprehensive_data.get('progress_ms'),
                    'duration_ms': comprehensive_data.get('duration_ms'),
                    'album_name': comprehensive_data.get('album_name'),
                    'popularity': comprehensive_data.get('popularity')
                }
                
                import json
                print(json.dumps(data_summary, indent=2))
                
                # Save to database
                if self.database.save_track_data(comprehensive_data):
                    print(f"💾 Saved to database")
                    
                    # Perform audio analysis for new tracks
                    if (current_track_id and 
                        current_track_id not in self.analysis_futures and
                        comprehensive_data.get('track_name') and 
                        comprehensive_data.get('primary_artist')):
                        
                        print(f"[{timestamp}] 🎵 Starting audio analysis...")
                        
                        # Submit analysis task to thread pool
                        track_name = comprehensive_data.get('track_name')
                        artist_name = comprehensive_data.get('primary_artist')
                        
                        future = self.audio_analyzer.submit_analysis_task(
                            track_name, 
                            artist_name, 
                            current_track_id,
                            self._update_analysis_status
                        )
                        
                        if future:
                            # Track the future for cleanup
                            self.analysis_futures[current_track_id] = future
                            print(f"[{timestamp}] 📊 Analysis task submitted (active tasks: {self.audio_analyzer.get_active_tasks_count()})")
                        else:
                            print(f"[{timestamp}] ℹ️  Track already analyzed or analysis skipped")
                        
                else:
                    print(f"❌ Failed to save to database")
            else:
                print(f"[{timestamp}] ℹ️  No data available")
            
            self.last_track_id = current_track_id
            
            # Clean up completed futures
            self._cleanup_completed_futures()
    
    def _update_analysis_status(self, track_id: str, status: str):
        """Callback to update analysis status in database."""
        try:
            self.database.update_analysis_status(track_id, status)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Analysis status: {track_id} -> {status}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to update status: {e}")
    
    def _cleanup_completed_futures(self):
        """Clean up completed analysis futures and update database with results."""
        completed_futures = []
        
        for track_id, future in self.analysis_futures.items():
            if future and future.done():
                try:
                    features = future.result(timeout=1)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Future result for {track_id}: {features is not None}")
                    
                    if features:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Features extracted: {list(features.keys())}")
                        # Update database with audio features
                        if self.database.update_track_audio_analysis(track_id, features):
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Audio features saved to database for {track_id}")
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to save audio features for {track_id}")
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  No features returned for {track_id}")
                        
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Analysis future error for {track_id}: {e}")
                
                completed_futures.append(track_id)
        
        # Remove completed futures
        for track_id in completed_futures:
            del self.analysis_futures[track_id]
    
    def start_tracking(self):
        """Start the continuous tracking process."""
        print("🎵 Starting Spotify Continuous Tracker...")
        print("=" * 50)
        
        # Test initial connection
        try:
            profile = self.client.get_user_profile()
            if profile:
                print(f"👤 Connected as: {profile.get('display_name', 'Unknown')}")
            else:
                print("❌ Failed to connect to Spotify API")
                return
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return
        
        # Show database stats
        try:
            stats = self.database.get_listening_stats()
            print(f"📊 Database stats: {stats['total_tracks']} tracks, {stats['unique_tracks']} unique, {stats['total_hours']} hours")
        except Exception as e:
            print(f"⚠️  Could not load database stats: {e}")
        
        self.running = True
        
        # Start token refresh worker in background
        token_thread = threading.Thread(target=self._token_refresh_worker, daemon=True)
        token_thread.start()
        
        print("🔄 Token refresh worker started")
        print("📡 Starting track polling (5-second interval)")
        print("=" * 50)
        
        # Main tracking loop
        try:
            while self.running:
                try:
                    # Get currently playing track
                    track_data = self.client.get_currently_playing()
                    
                    # Log track information
                    self._log_track_change(track_data)
                    
                    # Wait 5 seconds before next poll
                    time.sleep(5)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Polling error: {e}")
                    time.sleep(5)  # Continue polling even after errors
                    
        except KeyboardInterrupt:
            pass
        
        self.running = False
        print("\n🛑 Tracker stopped")

def main():
    """Main entry point."""
    tracker = SpotifyTracker()
    tracker.start_tracking()

if __name__ == "__main__":
    main() 
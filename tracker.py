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
    """Continuous Spotify tracking with automatic token refresh and audio analysis."""
    
    def __init__(self):
        self.client = SpotifyAPIClient()
        self.database = SupabaseDatabase()
        self.audio_analyzer = AudioAnalyzer()
        self.running = False
        self.last_track_id = None
        self.analyzed_tracks = set()  # Track which songs we've already analyzed
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
    
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
                        current_track_id not in self.analyzed_tracks and
                        comprehensive_data.get('track_name') and 
                        comprehensive_data.get('primary_artist')):
                        
                        print(f"[{timestamp}] 🎵 Starting audio analysis...")
                        
                        # Run audio analysis in a separate thread to avoid blocking
                        def analyze_audio():
                            try:
                                track_name = comprehensive_data.get('track_name')
                                artist_name = comprehensive_data.get('primary_artist')
                                
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎵 Analyzing: {track_name} by {artist_name}")
                                
                                # Download and analyze the track
                                audio_features = self.audio_analyzer.analyze_track(track_name, artist_name)
                                
                                if audio_features:
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Audio analysis completed:")
                                    for key, value in audio_features.items():
                                        print(f"    {key}: {value}")
                                    
                                    # Update the database with audio features
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Updating database for track: {current_track_id}")
                                    if self.database.update_track_audio_analysis(current_track_id, audio_features):
                                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Audio features saved to database successfully!")
                                        self.analyzed_tracks.add(current_track_id)
                                    else:
                                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed to save audio features to database")
                                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Track ID: {current_track_id}")
                                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Features: {audio_features}")
                                else:
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Audio analysis failed")
                                    
                            except Exception as e:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Audio analysis error: {e}")
                        
                        # Start audio analysis in background thread
                        analysis_thread = threading.Thread(target=analyze_audio, daemon=True)
                        analysis_thread.start()
                        
                else:
                    print(f"❌ Failed to save to database")
            else:
                print(f"[{timestamp}] ℹ️  No data available")
            
            self.last_track_id = current_track_id
    
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
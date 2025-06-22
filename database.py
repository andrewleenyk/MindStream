import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SpotifyDatabase:
    """SQLite database for storing Spotify track data."""
    
    def __init__(self, db_path: str = "data/spotify_tracks.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tracks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tracks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        track_id TEXT NOT NULL,
                        track_name TEXT NOT NULL,
                        primary_artist TEXT,
                        is_playing BOOLEAN,
                        progress_ms INTEGER,
                        duration_ms INTEGER,
                        album_name TEXT,
                        album_id TEXT,
                        popularity INTEGER,
                        explicit BOOLEAN,
                        track_number INTEGER,
                        disc_number INTEGER,
                        release_date TEXT,
                        album_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create artists table for many-to-many relationship
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS track_artists (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        track_id TEXT NOT NULL,
                        artist_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_timestamp ON tracks(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_track_id ON tracks(track_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(primary_artist)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_artists_track_id ON track_artists(track_id)')
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_track_data(self, track_data: Dict[str, Any]) -> bool:
        """
        Save track data to the database.
        
        Args:
            track_data: Dictionary containing track information
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert main track data
                cursor.execute('''
                    INSERT INTO tracks (
                        timestamp, track_id, track_name, primary_artist, is_playing,
                        progress_ms, duration_ms, album_name, album_id, popularity,
                        explicit, track_number, disc_number, release_date, album_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    track_data.get('timestamp'),
                    track_data.get('track_id'),
                    track_data.get('track_name'),
                    track_data.get('primary_artist'),
                    track_data.get('is_playing'),
                    track_data.get('progress_ms'),
                    track_data.get('duration_ms'),
                    track_data.get('album_name'),
                    track_data.get('album_id'),
                    track_data.get('popularity'),
                    track_data.get('explicit'),
                    track_data.get('track_number'),
                    track_data.get('disc_number'),
                    track_data.get('release_date'),
                    track_data.get('album_type')
                ))
                
                # Insert artists (many-to-many relationship)
                track_id = track_data.get('track_id')
                artists = track_data.get('artists', [])
                
                for artist_name in artists:
                    cursor.execute('''
                        INSERT INTO track_artists (track_id, artist_name)
                        VALUES (?, ?)
                    ''', (track_id, artist_name))
                
                conn.commit()
                logger.info(f"Saved track data for {track_data.get('track_name')}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save track data: {e}")
            return False
    
    def get_recent_tracks(self, limit: int = 10) -> list:
        """Get recent tracks from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM tracks 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                columns = [description[0] for description in cursor.description]
                tracks = []
                
                for row in cursor.fetchall():
                    track = dict(zip(columns, row))
                    tracks.append(track)
                
                return tracks
                
        except Exception as e:
            logger.error(f"Failed to get recent tracks: {e}")
            return []
    
    def get_tracks_by_artist(self, artist_name: str, limit: int = 50) -> list:
        """Get tracks by a specific artist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM tracks 
                    WHERE primary_artist LIKE ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (f'%{artist_name}%', limit))
                
                columns = [description[0] for description in cursor.description]
                tracks = []
                
                for row in cursor.fetchall():
                    track = dict(zip(columns, row))
                    tracks.append(track)
                
                return tracks
                
        except Exception as e:
            logger.error(f"Failed to get tracks by artist: {e}")
            return []
    
    def get_listening_stats(self) -> Dict[str, Any]:
        """Get basic listening statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total tracks
                cursor.execute('SELECT COUNT(*) FROM tracks')
                total_tracks = cursor.fetchone()[0]
                
                # Unique tracks
                cursor.execute('SELECT COUNT(DISTINCT track_id) FROM tracks')
                unique_tracks = cursor.fetchone()[0]
                
                # Top artists
                cursor.execute('''
                    SELECT primary_artist, COUNT(*) as play_count 
                    FROM tracks 
                    WHERE primary_artist IS NOT NULL 
                    GROUP BY primary_artist 
                    ORDER BY play_count DESC 
                    LIMIT 5
                ''')
                top_artists = cursor.fetchall()
                
                # Total listening time (approximate)
                cursor.execute('SELECT SUM(duration_ms) FROM tracks WHERE duration_ms > 0')
                total_ms = cursor.fetchone()[0] or 0
                total_hours = total_ms / (1000 * 60 * 60)
                
                return {
                    'total_tracks': total_tracks,
                    'unique_tracks': unique_tracks,
                    'top_artists': top_artists,
                    'total_hours': round(total_hours, 2)
                }
                
        except Exception as e:
            logger.error(f"Failed to get listening stats: {e}")
            return {}

if __name__ == "__main__":
    # Test the database
    db = SpotifyDatabase()
    print("✅ Database initialized successfully!")
    
    # Test stats
    stats = db.get_listening_stats()
    print(f"📊 Current stats: {stats}") 
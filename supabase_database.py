import psycopg2
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SupabaseDatabase:
    """Supabase PostgreSQL database for storing Spotify track data."""
    
    def __init__(self):
        """Initialize the database connection."""
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish connection to Supabase PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('SUPABASE_HOST'),
                port=os.getenv('SUPABASE_PORT', 5432),
                dbname=os.getenv('SUPABASE_DB'),
                user=os.getenv('SUPABASE_USER'),
                password=os.getenv('SUPABASE_PASSWORD')
            )
            self.conn.autocommit = True
            logger.info("Connected to Supabase database successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase database: {e}")
            raise
    
    def save_track_data(self, track_data: Dict[str, Any]) -> bool:
        """
        Save track data to the Supabase database.
        
        Args:
            track_data: Dictionary containing track information
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                # Insert main track data
                cur.execute('''
                    INSERT INTO tracks (
                        timestamp, track_id, track_name, primary_artist, artists,
                        is_playing, progress_ms, duration_ms, album_name, album_id,
                        popularity, explicit, track_number, disc_number, release_date, album_type
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    track_data.get('timestamp'),
                    track_data.get('track_id'),
                    track_data.get('track_name'),
                    track_data.get('primary_artist'),
                    track_data.get('artists', []),  # PostgreSQL array
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
                
                logger.info(f"Saved track data for {track_data.get('track_name')}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save track data: {e}")
            return False
    
    def get_recent_tracks(self, limit: int = 10) -> list:
        """Get recent tracks from the database."""
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                cur.execute('''
                    SELECT * FROM tracks 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                ''', (limit,))
                
                columns = [description[0] for description in cur.description]
                tracks = []
                
                for row in cur.fetchall():
                    track = dict(zip(columns, row))
                    tracks.append(track)
                
                return tracks
                
        except Exception as e:
            logger.error(f"Failed to get recent tracks: {e}")
            return []
    
    def get_tracks_by_artist(self, artist_name: str, limit: int = 50) -> list:
        """Get tracks by a specific artist."""
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                cur.execute('''
                    SELECT * FROM tracks 
                    WHERE primary_artist ILIKE %s 
                    ORDER BY timestamp DESC 
                    LIMIT %s
                ''', (f'%{artist_name}%', limit))
                
                columns = [description[0] for description in cur.description]
                tracks = []
                
                for row in cur.fetchall():
                    track = dict(zip(columns, row))
                    tracks.append(track)
                
                return tracks
                
        except Exception as e:
            logger.error(f"Failed to get tracks by artist: {e}")
            return []
    
    def get_listening_stats(self) -> Dict[str, Any]:
        """Get basic listening statistics."""
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                # Total tracks
                cur.execute('SELECT COUNT(*) FROM tracks')
                total_tracks = cur.fetchone()[0]
                
                # Unique tracks
                cur.execute('SELECT COUNT(DISTINCT track_id) FROM tracks')
                unique_tracks = cur.fetchone()[0]
                
                # Top artists
                cur.execute('''
                    SELECT primary_artist, COUNT(*) as play_count 
                    FROM tracks 
                    WHERE primary_artist IS NOT NULL 
                    GROUP BY primary_artist 
                    ORDER BY play_count DESC 
                    LIMIT 5
                ''')
                top_artists = cur.fetchall()
                
                # Total listening time (approximate)
                cur.execute('SELECT SUM(duration_ms) FROM tracks WHERE duration_ms > 0')
                total_ms = cur.fetchone()[0] or 0
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
    
    def close(self):
        """Close the database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    # Test the Supabase connection
    try:
        db = SupabaseDatabase()
        print("✅ Connected to Supabase database successfully!")
        
        # Test stats
        stats = db.get_listening_stats()
        print(f"📊 Current stats: {stats}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}") 
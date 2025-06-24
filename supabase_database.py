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
        # Ensure enhanced Mozart feature columns exist
        self.ensure_enhanced_features_columns()
    
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
            track_data: Dictionary containing track information and optional audio analysis
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                # Insert main track data (removed unused columns)
                cur.execute('''
                    INSERT INTO tracks (
                        timestamp, track_id, track_name, primary_artist, artists,
                        is_playing, progress_ms, duration_ms, album_name, album_id,
                        popularity, explicit, track_number, disc_number, release_date, album_type,
                        tempo, valence, key, mode, energy, loudness, instrumentalness, 
                        acousticness, speechiness, danceability, analysis_status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    track_data.get('album_type'),
                    track_data.get('tempo'),
                    track_data.get('valence'),
                    track_data.get('key'),
                    track_data.get('mode'),
                    track_data.get('energy'),
                    track_data.get('loudness'),
                    track_data.get('instrumentalness'),
                    track_data.get('acousticness'),
                    track_data.get('speechiness'),
                    track_data.get('danceability'),
                    track_data.get('analysis_status', 'pending')  # Default to pending
                ))
                
                logger.info(f"Saved track data for {track_data.get('track_name')}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save track data: {e}")
            return False
    
    def update_track_audio_analysis(self, track_id: str, audio_features: Dict[str, Any]) -> bool:
        """
        Update an existing track record with audio analysis data.
        Now supports enhanced Mozart features including confidence scores.
        
        Args:
            track_id: Spotify track ID
            audio_features: Dictionary containing audio analysis features
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                # Check if we have enhanced Mozart features
                has_enhanced_features = any(key.endswith('_confidence') for key in audio_features.keys())
                
                if has_enhanced_features:
                    # Use enhanced Mozart features
                    cur.execute('''
                        UPDATE tracks 
                        SET tempo = %s, valence = %s, danceability = %s, instrumentalness = %s, 
                            acousticness = %s, speechiness = %s, key = %s, mode = %s,
                            energy = %s, loudness = %s,
                            valence_confidence = %s, danceability_confidence = %s, 
                            instrumentalness_confidence = %s, acousticness_confidence = %s, 
                            speechiness_confidence = %s,
                            analysis_status = %s
                        WHERE track_id = %s
                    ''', (
                        audio_features.get('tempo'),
                        audio_features.get('valence'),
                        audio_features.get('danceability'),
                        audio_features.get('instrumentalness'),
                        audio_features.get('acousticness'),
                        audio_features.get('speechiness'),
                        audio_features.get('key'),
                        audio_features.get('mode'),
                        audio_features.get('energy'),
                        audio_features.get('loudness'),
                        audio_features.get('valence_confidence'),
                        audio_features.get('danceability_confidence'),
                        audio_features.get('instrumentalness_confidence'),
                        audio_features.get('acousticness_confidence'),
                        audio_features.get('speechiness_confidence'),
                        'completed',  # Set analysis status to completed
                        track_id
                    ))
                else:
                    # Use basic features (backward compatibility)
                    cur.execute('''
                        UPDATE tracks 
                        SET tempo = %s, valence = %s, key = %s, mode = %s,
                            energy = %s, loudness = %s, instrumentalness = %s, acousticness = %s, speechiness = %s, danceability = %s,
                            analysis_status = %s
                        WHERE track_id = %s
                    ''', (
                        audio_features.get('tempo'),
                        audio_features.get('valence'),
                        audio_features.get('key'),
                        audio_features.get('mode'),
                        audio_features.get('energy'),
                        audio_features.get('loudness'),
                        audio_features.get('instrumentalness'),
                        audio_features.get('acousticness'),
                        audio_features.get('speechiness'),
                        audio_features.get('danceability'),
                        'completed',  # Set analysis status to completed
                        track_id
                    ))
                
                if cur.rowcount > 0:
                    feature_type = "enhanced Mozart" if has_enhanced_features else "basic"
                    logger.info(f"Updated {feature_type} audio analysis for track {track_id}")
                    return True
                else:
                    logger.warning(f"No track found with ID {track_id} to update")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update audio analysis: {e}")
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
    
    def update_analysis_status(self, track_id: str, status: str) -> bool:
        """
        Update the analysis status for a track.
        
        Args:
            track_id: Spotify track ID
            status: Status string ('pending', 'in_progress', 'completed', 'failed')
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                cur.execute('''
                    UPDATE tracks 
                    SET analysis_status = %s
                    WHERE track_id = %s
                ''', (status, track_id))
                
                if cur.rowcount > 0:
                    logger.info(f"Updated analysis status to '{status}' for track {track_id}")
                    return True
                else:
                    logger.warning(f"No track found with ID {track_id} to update status")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update analysis status: {e}")
            return False
    
    def get_tracks_needing_analysis(self, limit: int = 10) -> list:
        """
        Get tracks that need audio analysis (pending or failed status).
        
        Args:
            limit: Maximum number of tracks to return
            
        Returns:
            List of track dictionaries
        """
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                cur.execute('''
                    SELECT track_id, track_name, primary_artist, analysis_status
                    FROM tracks 
                    WHERE analysis_status IN ('pending', 'failed')
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
            logger.error(f"Failed to get tracks needing analysis: {e}")
            return []
    
    def has_completed_analysis(self, track_id: str) -> bool:
        """Return True if the track has completed analysis in the database."""
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            with self.conn.cursor() as cur:
                # Check for completed status OR tracks with analysis data (tempo is not null)
                cur.execute('''
                    SELECT 1 FROM tracks 
                    WHERE track_id = %s 
                    AND (analysis_status = 'completed' OR tempo IS NOT NULL)
                    LIMIT 1
                ''', (track_id,))
                return cur.fetchone() is not None
        except Exception as e:
            print(f"❌ Error checking analysis status: {e}")
            return False
    
    def track_exists(self, track_id: str) -> bool:
        """Return True if the track exists in the database at all."""
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            with self.conn.cursor() as cur:
                cur.execute('''
                    SELECT 1 FROM tracks 
                    WHERE track_id = %s 
                    LIMIT 1
                ''', (track_id,))
                return cur.fetchone() is not None
        except Exception as e:
            print(f"❌ Error checking if track exists: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed")
    
    def get_existing_analysis_data(self, track_id: str) -> Dict[str, Any]:
        """Get existing analysis data for a track from any previous occurrence."""
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            with self.conn.cursor() as cur:
                cur.execute('''
                    SELECT tempo, beat_strength, rhythmic_stability, regularity, 
                           valence, key, mode, energy, loudness, instrumentalness, 
                           acousticness, speechiness, danceability
                    FROM tracks 
                    WHERE track_id = %s 
                    AND (analysis_status = 'completed' OR tempo IS NOT NULL)
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (track_id,))
                
                row = cur.fetchone()
                if row:
                    columns = ['tempo', 'beat_strength', 'rhythmic_stability', 'regularity', 
                              'valence', 'key', 'mode', 'energy', 'loudness', 'instrumentalness', 
                              'acousticness', 'speechiness', 'danceability']
                    return dict(zip(columns, row))
                return {}
        except Exception as e:
            print(f"❌ Error fetching existing analysis data: {e}")
            return {}
    
    def update_current_entry_with_existing_analysis(self, track_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Update the most recent track entry with existing analysis data."""
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                # Update the most recent entry for this track
                cur.execute('''
                    UPDATE tracks 
                    SET tempo = %s, beat_strength = %s, rhythmic_stability = %s, 
                        regularity = %s, valence = %s, key = %s, mode = %s,
                        energy = %s, loudness = %s, instrumentalness = %s, acousticness = %s, speechiness = %s, danceability = %s,
                        analysis_status = %s
                    WHERE track_id = %s 
                    AND timestamp = (
                        SELECT MAX(timestamp) 
                        FROM tracks 
                        WHERE track_id = %s
                    )
                ''', (
                    analysis_data.get('tempo'),
                    analysis_data.get('beat_strength'),
                    analysis_data.get('rhythmic_stability'),
                    analysis_data.get('regularity'),
                    analysis_data.get('valence'),
                    analysis_data.get('key'),
                    analysis_data.get('mode'),
                    analysis_data.get('energy'),
                    analysis_data.get('loudness'),
                    analysis_data.get('instrumentalness'),
                    analysis_data.get('acousticness'),
                    analysis_data.get('speechiness'),
                    analysis_data.get('danceability'),
                    'completed',
                    track_id,
                    track_id
                ))
                
                if cur.rowcount > 0:
                    print(f"✅ Updated current entry with existing analysis data for {track_id}")
                    return True
                else:
                    print(f"❌ No current entry found to update for {track_id}")
                    return False
                
        except Exception as e:
            print(f"❌ Error updating current entry with analysis: {e}")
            return False
    
    def ensure_enhanced_features_columns(self):
        """
        Ensure the database table has the enhanced Mozart feature columns.
        Adds confidence columns if they don't exist.
        """
        try:
            if not self.conn or self.conn.closed:
                self.connect()
            
            with self.conn.cursor() as cur:
                # Check if confidence columns exist
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'tracks' 
                    AND column_name IN ('valence_confidence', 'danceability_confidence', 'instrumentalness_confidence', 'acousticness_confidence', 'speechiness_confidence')
                """)
                
                existing_columns = [row[0] for row in cur.fetchall()]
                missing_columns = []
                
                required_columns = [
                    'valence_confidence',
                    'danceability_confidence', 
                    'instrumentalness_confidence',
                    'acousticness_confidence',
                    'speechiness_confidence'
                ]
                
                for col in required_columns:
                    if col not in existing_columns:
                        missing_columns.append(col)
                
                # Add missing columns
                for col in missing_columns:
                    cur.execute(f"ALTER TABLE tracks ADD COLUMN {col} FLOAT")
                    logger.info(f"Added column {col} to tracks table")
                
                if missing_columns:
                    logger.info(f"Enhanced Mozart feature columns added: {missing_columns}")
                else:
                    logger.info("All enhanced Mozart feature columns already exist")
                    
        except Exception as e:
            logger.error(f"Failed to ensure enhanced features columns: {e}")
            # Don't raise - this is not critical for basic functionality

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
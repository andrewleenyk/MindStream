import requests
import json
from typing import Optional, Dict, Any
import logging

# Import our auth module from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auth import get_spotify_headers

# Set up logging
logger = logging.getLogger(__name__)

class SpotifyAPIClient:
    """Spotify API Client for fetching user data and currently playing track."""
    
    def __init__(self):
        """Initialize the Spotify API client."""
        self.base_url = "https://api.spotify.com/v1"
        self.headers = None  # Will be set when we make our first request
        
    def _get_headers(self):
        """Get fresh headers with current access token."""
        try:
            return get_spotify_headers()
        except Exception as e:
            logger.error(f"Failed to get headers: {e}")
            return None
    
    def _make_request(self, endpoint: str, method: str = "GET") -> Optional[Dict[str, Any]]:
        """
        Make a request to the Spotify API.
        
        Args:
            endpoint: The API endpoint (e.g., "/me/player/currently-playing")
            method: HTTP method (default: "GET")
            
        Returns:
            JSON response from Spotify API or None if error
        """
        try:
            headers = self._get_headers()
            if not headers:
                logger.error("Failed to get authentication headers")
                return None
            
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            # Handle different status codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                logger.error("Authentication failed - token may be invalid")
                return None
            elif response.status_code == 429:
                logger.error("Rate limit exceeded - too many requests")
                return None
            elif response.status_code == 404:
                logger.error(f"Endpoint not found: {endpoint}")
                return None
            else:
                logger.error(f"HTTP {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
    
    def get_currently_playing(self) -> Optional[Dict[str, Any]]:
        """Get the currently playing track from Spotify."""
        endpoint = "/me/player/currently-playing"
        response = self._make_request(endpoint)
        
        if response is None:
            logger.info("No response from Spotify API - user may not be playing anything")
            return None
        
        # Spotify returns 204 (No Content) when nothing is playing
        # But our _make_request handles this by returning None
        if not response:
            logger.info("Nothing currently playing")
            return None
        
        logger.info("Successfully retrieved currently playing track")
        return response
    
    def get_user_profile(self) -> Optional[Dict[str, Any]]:
        """Get the current user's profile information."""
        endpoint = "/me"
        response = self._make_request(endpoint)
        
        if response is None:
            logger.error("Failed to get user profile")
            return None
        
        logger.info("Successfully retrieved user profile")
        return response
    
    def get_comprehensive_track_data(self, track_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive track data from the currently-playing response.
        
        Args:
            track_data: Raw response from currently-playing endpoint
            
        Returns:
            Combined dictionary with all track data
        """
        if not track_data or 'item' not in track_data:
            return None
        
        item = track_data['item']
        track_id = item.get('id')
        
        if not track_id:
            logger.error("No track ID found in response")
            return None
        
        # Extract comprehensive track data
        comprehensive_data = {
            'timestamp': track_data.get('timestamp'),
            'track_id': track_id,
            'track_name': item.get('name'),
            'primary_artist': item.get('artists', [{}])[0].get('name') if item.get('artists') else None,
            'is_playing': track_data.get('is_playing', False),
            'progress_ms': track_data.get('progress_ms', 0),
            'duration_ms': item.get('duration_ms', 0),
            'album_name': item.get('album', {}).get('name'),
            'album_id': item.get('album', {}).get('id'),
            'artists': [artist.get('name') for artist in item.get('artists', [])],
            'external_urls': item.get('external_urls', {}),
            'popularity': item.get('popularity'),
            'explicit': item.get('explicit', False),
            'track_number': item.get('track_number'),
            'disc_number': item.get('disc_number'),
            'release_date': item.get('album', {}).get('release_date'),
            'album_type': item.get('album', {}).get('album_type'),
            'available_markets': item.get('available_markets', [])
        }
        
        return comprehensive_data

def test_currently_playing():
    """Test function to check if we can get currently playing track."""
    try:
        print("🎵 Testing Spotify API Client...")
        
        # Create client instance
        client = SpotifyAPIClient()
        
        # Test user profile first
        print("\n👤 Testing user profile...")
        profile = client.get_user_profile()
        if profile:
            print(f"✅ User: {profile.get('display_name', 'Unknown')}")
            print(f"   ID: {profile.get('id', 'Unknown')}")
        else:
            print("❌ Failed to get user profile")
            return
        
        # Test currently playing
        print("\n🎧 Testing currently playing...")
        current_track = client.get_currently_playing()
        
        if current_track:
            item = current_track.get('item', {})
            if item:
                track_name = item.get('name', 'Unknown Track')
                artists = [artist.get('name', 'Unknown') for artist in item.get('artists', [])]
                artist_names = ', '.join(artists)
                
                print(f"✅ Now Playing: {track_name} by {artist_names}")
                print(f"   Album: {item.get('album', {}).get('name', 'Unknown')}")
                print(f"   Duration: {current_track.get('progress_ms', 0) // 1000}s / {item.get('duration_ms', 0) // 1000}s")
            else:
                print("ℹ️  Track data not available")
        else:
            print("ℹ️  Nothing currently playing")
            
        print("\n🎉 Spotify API Client test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    # Run the test when this file is executed directly
    test_currently_playing()

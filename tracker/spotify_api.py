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
    """
    Spotify API Client for fetching user data and currently playing track.
    
    This class handles all interactions with the Spotify Web API.
    """
    
    def __init__(self):
        """Initialize the Spotify API client."""
        self.base_url = "https://api.spotify.com/v1"
        self.headers = None  # Will be set when we make our first request
        
    def _get_headers(self):
        """
        Get fresh headers with current access token.
        
        TODO: This function should:
        1. Call get_spotify_headers() from our auth module
        2. Return the headers for API requests
        3. Handle any authentication errors gracefully
        """
        try:
            return get_spotify_headers()
        except Exception as e:
            logger.error(f"Failed to get headers: {e}")
            return None
    
    def _make_request(self, endpoint: str, method: str = "GET") -> Optional[Dict[str, Any]]:
        """
        Make a request to the Spotify API.
        
        TODO: This function should:
        1. Get fresh headers using _get_headers()
        2. Make the HTTP request to the Spotify API
        3. Handle different HTTP status codes:
           - 200: Return the JSON response
           - 401: Log authentication error
           - 429: Handle rate limiting
           - 404: Handle not found
           - Other errors: Log and return None
        4. Return the JSON response or None if error
        
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
        """
        Get the currently playing track from Spotify.
        
        TODO: This function should:
        1. Call _make_request() with the currently-playing endpoint
        2. Handle the case where nothing is currently playing
        3. Return the track data or None if nothing is playing
        
        Returns:
            Dictionary with track information or None if nothing is playing
        """
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
        """
        Get the current user's profile information.
        
        TODO: This function should:
        1. Call _make_request() with the user profile endpoint
        2. Return user profile data
        
        Returns:
            Dictionary with user profile information
        """
        endpoint = "/me"
        response = self._make_request(endpoint)
        
        if response is None:
            logger.error("Failed to get user profile")
            return None
        
        logger.info("Successfully retrieved user profile")
        return response

def test_currently_playing():
    """
    Test function to check if we can get currently playing track.
    
    TODO: This function should:
    1. Create a SpotifyAPIClient instance
    2. Call get_currently_playing()
    3. Print the result to console
    4. Handle any errors gracefully
    """
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

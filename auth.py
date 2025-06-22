import os
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenManager:
    def __init__(self):
        self.access_token = None
        self.refresh_token = os.getenv('SPOTIFY_REFRESH_TOKEN')
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.token_expiry = None
        self.refresh_interval = 55 * 60  # 55 minutes in seconds
        
        if not self.refresh_token:
            raise ValueError("SPOTIFY_REFRESH_TOKEN not found in environment variables")
        if not self.client_id:
            raise ValueError("SPOTIFY_CLIENT_ID not found in environment variables")
        if not self.client_secret:
            raise ValueError("SPOTIFY_CLIENT_SECRET not found in environment variables")
    
    def refresh_access_token(self):
        """Refresh the access token using the refresh token"""
        try:
            url = "https://accounts.spotify.com/api/token"
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }
            
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            # Set expiry time (Spotify tokens typically last 1 hour)
            self.token_expiry = datetime.now() + timedelta(hours=1)
            
            # Update refresh token if a new one is provided
            if 'refresh_token' in token_data:
                self.refresh_token = token_data['refresh_token']
                logger.info("New refresh token received")
            
            logger.info(f"Access token refreshed successfully. Expires at: {self.token_expiry}")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to refresh access token: {e}")
            raise
    
    def get_access_token(self):
        """Get a valid access token, refreshing if necessary"""
        current_time = datetime.now()
        
        # If no token or token is expired/expiring soon, refresh it
        if (not self.access_token or 
            not self.token_expiry or 
            current_time >= self.token_expiry - timedelta(minutes=5)):
            
            logger.info("Access token needs refresh")
            return self.refresh_access_token()
        
        return self.access_token
    
    def start_auto_refresh(self):
        """Start automatic token refresh every 55 minutes"""
        logger.info("Starting automatic token refresh every 55 minutes")
        
        while True:
            try:
                # Get a fresh token
                self.get_access_token()
                
                # Wait for 55 minutes before next refresh
                time.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                logger.info("Auto-refresh stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in auto-refresh loop: {e}")
                # Wait a bit before retrying
                time.sleep(60)

# Global token manager instance
token_manager = TokenManager()

def get_spotify_headers():
    """Get headers with current access token for Spotify API requests"""
    return {
        "Authorization": f"Bearer {token_manager.get_access_token()}",
        "Content-Type": "application/json"
    }

if __name__ == "__main__":
    # Test the token refresh
    try:
        token = token_manager.get_access_token()
        print(f"Successfully obtained access token: {token[:20]}...")
        
        # Uncomment the line below to start auto-refresh
        # token_manager.start_auto_refresh()
        
    except Exception as e:
        print(f"Error: {e}") 
# Mindstream - Spotify Continuous Tracker

A Python application that continuously tracks your Spotify listening activity with automatic token refresh.

## Features

- 🎵 **Continuous tracking** - Polls Spotify API every 5 seconds
- 🔄 **Automatic token refresh** - Refreshes access tokens every 55 minutes
- 📊 **Comprehensive data collection** - Extracts detailed track information
- 🛡️ **Robust error handling** - Graceful handling of network issues and API errors
- 📝 **Real-time logging** - Console output with timestamps and track changes

## Quick Start

1. **Clone the repository**

   ```bash
   git clone git@github.com:andrewleenyk/Mindstream.git
   cd Mindstream
   ```

2. **Set up virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file with your Spotify credentials:

   ```
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   SPOTIFY_REFRESH_TOKEN=your_refresh_token_here
   ```

4. **Run the tracker**
   ```bash
   python tracker.py
   ```

## Project Structure

```
Mindstream/
├── auth.py                 # Token management and refresh
├── tracker.py              # Main continuous tracker script
├── tracker/
│   ├── __init__.py
│   └── spotify_api.py      # Spotify API client
├── data/                   # Data storage directory
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (not tracked)
```

## Data Collected

For each track, the system collects:

- **Basic info**: Track name, artist, album, duration
- **Playback state**: Currently playing, progress, timestamp
- **Metadata**: Popularity, explicit content, release date
- **IDs**: Track ID, album ID, artist IDs
- **External links**: Spotify URLs

## Usage

The tracker runs continuously and will:

- Display track changes in real-time
- Show comprehensive data for each track
- Automatically handle token refresh
- Gracefully handle errors and network issues

Press `Ctrl+C` to stop the tracker.

## Development

- **Test API client**: `python tracker/spotify_api.py`
- **Test authentication**: `python auth.py`

## Requirements

- Python 3.7+
- Spotify Developer Account
- Valid Spotify API credentials

## License

MIT License

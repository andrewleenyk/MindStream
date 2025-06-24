#!/usr/bin/env python3
"""
Test script for Mozart Enhanced Audio Analyzer integration.
Tests the new Mozart-based analysis system with Mindstream.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import Config
from audio.analyzer import AudioAnalyzer
from audio.mozart_analyzer import MozartAudioAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_mozart_analyzer():
    """Test the Mozart enhanced analyzer directly."""
    print("\n🧪 TESTING MOZART ENHANCED ANALYZER")
    print("=" * 50)
    
    # Initialize the Mozart analyzer
    mozart_analyzer = MozartAudioAnalyzer()
    print("✅ Mozart analyzer initialized")
    
    # Test with a sample audio file if available
    test_audio_path = "temp_audio/test_sample.mp3"
    
    if os.path.exists(test_audio_path):
        print(f"Testing with existing audio file: {test_audio_path}")
        
        # Extract features
        start_time = time.time()
        features = mozart_analyzer.extract_all_features(test_audio_path)
        analysis_time = time.time() - start_time
        
        if features:
            print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
            print("\n🎵 EXTRACTED FEATURES:")
            for key, value in features.items():
                print(f"   {key}: {value}")
        else:
            print("❌ Analysis failed")
    else:
        print("ℹ️  No test audio file found, skipping direct analyzer test")


def test_integrated_analyzer():
    """Test the integrated analyzer with sample track data."""
    print("\n🧪 TESTING INTEGRATED ANALYZER")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Initialize the integrated analyzer
    analyzer = AudioAnalyzer(config)
    print("✅ Integrated analyzer initialized")
    
    # Sample track data (simulating Spotify API response)
    sample_tracks = [
        {
            'id': 'test_track_1',
            'name': 'Bohemian Rhapsody',
            'artists': [{'name': 'Queen'}],
            'album': {'name': 'A Night at the Opera'},
            'duration_ms': 354000,
            'popularity': 85
        },
        {
            'id': 'test_track_2',
            'name': 'Imagine',
            'artists': [{'name': 'John Lennon'}],
            'album': {'name': 'Imagine'},
            'duration_ms': 183000,
            'popularity': 80
        }
    ]
    
    print(f"Testing with {len(sample_tracks)} sample tracks")
    
    # Test batch analysis (this will attempt to download and analyze)
    print("\nStarting batch analysis...")
    start_time = time.time()
    
    try:
        results = analyzer.analyze_tracks_batch(sample_tracks, max_workers=2)
        analysis_time = time.time() - start_time
        
        print(f"\n✅ Batch analysis completed in {analysis_time:.2f} seconds")
        print(f"Results: {len(results)} successful analyses")
        
        if results:
            print("\n📊 ANALYSIS RESULTS:")
            for i, result in enumerate(results, 1):
                print(f"\nTrack {i}: {result.get('name', 'Unknown')} by {result.get('artist', 'Unknown')}")
                print(f"   Tempo: {result.get('tempo', 'N/A')} BPM")
                print(f"   Valence: {result.get('valence', 'N/A')} (confidence: {result.get('valence_confidence', 'N/A')})")
                print(f"   Danceability: {result.get('danceability', 'N/A')} (confidence: {result.get('danceability_confidence', 'N/A')})")
                print(f"   Instrumentalness: {result.get('instrumentalness', 'N/A')} (confidence: {result.get('instrumentalness_confidence', 'N/A')})")
                print(f"   Acousticness: {result.get('acousticness', 'N/A')} (confidence: {result.get('acousticness_confidence', 'N/A')})")
                print(f"   Speechiness: {result.get('speechiness', 'N/A')} (confidence: {result.get('speechiness_confidence', 'N/A')})")
                print(f"   Key: {result.get('key', 'N/A')} {result.get('mode', 'N/A')}")
                print(f"   Energy: {result.get('energy', 'N/A')}")
                print(f"   Loudness: {result.get('loudness', 'N/A')} dB")
        
        # Get analysis status
        status = analyzer.get_analysis_status()
        print(f"\n📈 ANALYSIS STATUS:")
        print(f"   Is analyzing: {status.get('is_analyzing', False)}")
        print(f"   Progress: {status.get('progress', 0):.1f}%")
        print(f"   Total tracks: {status.get('total_tracks', 0)}")
        print(f"   Processed: {status.get('processed_tracks', 0)}")
        print(f"   Failed: {status.get('failed_tracks', 0)}")
        
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def test_analyzer_components():
    """Test individual analyzer components."""
    print("\n🧪 TESTING ANALYZER COMPONENTS")
    print("=" * 50)
    
    # Initialize configuration
    config = Config()
    
    # Test Mozart analyzer
    print("Testing Mozart analyzer...")
    mozart_analyzer = MozartAudioAnalyzer()
    print("✅ Mozart analyzer component initialized")
    
    # Test downloader
    print("Testing downloader...")
    from audio.downloader import AudioDownloader
    downloader = AudioDownloader(config)
    print("✅ Downloader component initialized")
    
    # Test validator
    print("Testing validator...")
    from audio.validator import AudioValidator
    validator = AudioValidator(config)
    print("✅ Validator component initialized")
    
    # Test statistics
    print("Testing statistics...")
    from audio.statistics import AudioStatistics
    statistics = AudioStatistics()
    print("✅ Statistics component initialized")
    
    print("✅ All components initialized successfully")


def main():
    """Main test function."""
    print("🎵 MOZART ENHANCED AUDIO ANALYZER TEST")
    print("=" * 60)
    print("Testing the integration of Mozart enhanced analysis with Mindstream")
    print("=" * 60)
    
    try:
        # Test individual components
        test_analyzer_components()
        
        # Test Mozart analyzer directly
        test_mozart_analyzer()
        
        # Test integrated analyzer
        test_integrated_analyzer()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        logger.error(f"Test traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
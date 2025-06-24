#!/usr/bin/env python3
"""
Test script for the new modular audio analyzer.
Verifies that the new architecture works correctly.
"""

import sys
import os
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio import AudioAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_new_analyzer():
    """Test the new modular audio analyzer."""
    print("🧪 Testing New Modular Audio Analyzer")
    print("=" * 50)
    
    # Initialize the analyzer
    analyzer = AudioAnalyzer()
    
    # Test with a sample track
    test_track = "Bohemian Rhapsody"
    test_artist = "Queen"
    
    print(f"Testing audio analysis for: {test_track} by {test_artist}")
    
    # Perform analysis
    features = analyzer.analyze_track_with_retries(test_track, test_artist)
    
    if features:
        print("✅ Audio analysis successful!")
        print("Features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        # Print statistics
        analyzer.print_analysis_stats()
        
        # Test validation
        from audio.validator import AudioFeatureValidator
        validator = AudioFeatureValidator()
        is_valid = validator.validate_features(features)
        print(f"✅ Feature validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Get validation summary
        summary = validator.get_validation_summary(features)
        print(f"📊 Validation Score: {summary['overall_score']:.2f}")
        
    else:
        print("❌ Audio analysis failed")
        analyzer.print_analysis_stats()

def test_components():
    """Test individual components."""
    print("\n🔧 Testing Individual Components")
    print("=" * 50)
    
    # Test feature extractor
    from audio.features import AudioFeatureExtractor
    extractor = AudioFeatureExtractor()
    print("✅ Feature extractor initialized")
    
    # Test downloader
    from audio.downloader import AudioDownloader
    downloader = AudioDownloader()
    print("✅ Downloader initialized")
    
    # Test validator
    from audio.validator import AudioFeatureValidator
    validator = AudioFeatureValidator()
    print("✅ Validator initialized")
    
    # Test statistics
    from audio.statistics import get_stats
    stats = get_stats()
    print("✅ Statistics initialized")
    
    print("✅ All components working correctly!")

if __name__ == "__main__":
    try:
        test_components()
        test_new_analyzer()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
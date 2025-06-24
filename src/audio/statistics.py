"""
Audio Statistics Module for Mindstream
Generates statistics and insights from audio analysis results.
"""

import logging
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class AudioStatistics:
    """Generates statistics and insights from audio analysis results."""
    
    def __init__(self):
        """Initialize the audio statistics module."""
        logger.info("✅ Audio Statistics initialized")
    
    def generate_analysis_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive statistics from analysis results.
        
        Args:
            results: List of analysis result dictionaries
            
        Returns:
            Dictionary containing various statistics
        """
        if not results:
            return {'error': 'No analysis results provided'}
        
        try:
            stats = {
                'total_tracks': len(results),
                'feature_statistics': self._calculate_feature_statistics(results),
                'genre_insights': self._generate_genre_insights(results),
                'mood_analysis': self._generate_mood_analysis(results),
                'performance_metrics': self._calculate_performance_metrics(results),
                'quality_metrics': self._calculate_quality_metrics(results)
            }
            
            logger.info(f"Generated statistics for {len(results)} tracks")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_feature_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic statistics for each feature."""
        numeric_features = [
            'tempo', 'valence', 'danceability', 'instrumentalness',
            'acousticness', 'speechiness', 'energy', 'loudness'
        ]
        
        stats = {}
        for feature in numeric_features:
            values = [r.get(feature) for r in results if r.get(feature) is not None]
            if values:
                stats[feature] = {
                    'mean': round(np.mean(values), 3),
                    'median': round(np.median(values), 3),
                    'std': round(np.std(values), 3),
                    'min': round(np.min(values), 3),
                    'max': round(np.max(values), 3),
                    'count': len(values)
                }
        
        # Categorical features
        key_counts = defaultdict(int)
        mode_counts = defaultdict(int)
        
        for result in results:
            key = result.get('key', 'Unknown')
            mode = result.get('mode', 'Unknown')
            key_counts[key] += 1
            mode_counts[mode] += 1
        
        stats['key_distribution'] = dict(key_counts)
        stats['mode_distribution'] = dict(mode_counts)
        
        return stats
    
    def _generate_genre_insights(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights about musical characteristics that might indicate genres."""
        insights = {
            'high_energy_tracks': [],
            'acoustic_tracks': [],
            'danceable_tracks': [],
            'instrumental_tracks': [],
            'vocal_tracks': []
        }
        
        for result in results:
            name = result.get('name', 'Unknown')
            artist = result.get('artist', 'Unknown')
            
            # High energy tracks (potential rock, electronic)
            if result.get('energy', 0) > 0.8:
                insights['high_energy_tracks'].append({
                    'name': name,
                    'artist': artist,
                    'energy': result.get('energy', 0)
                })
            
            # Acoustic tracks
            if result.get('acousticness', 0) > 0.8:
                insights['acoustic_tracks'].append({
                    'name': name,
                    'artist': artist,
                    'acousticness': result.get('acousticness', 0)
                })
            
            # Danceable tracks
            if result.get('danceability', 0) > 0.8:
                insights['danceable_tracks'].append({
                    'name': name,
                    'artist': artist,
                    'danceability': result.get('danceability', 0)
                })
            
            # Instrumental tracks
            if result.get('instrumentalness', 0) > 0.8:
                insights['instrumental_tracks'].append({
                    'name': name,
                    'artist': artist,
                    'instrumentalness': result.get('instrumentalness', 0)
                })
            
            # Vocal tracks (low instrumentalness)
            if result.get('instrumentalness', 0) < 0.2:
                insights['vocal_tracks'].append({
                    'name': name,
                    'artist': artist,
                    'instrumentalness': result.get('instrumentalness', 0)
                })
        
        # Sort by feature values
        for category in insights:
            insights[category] = sorted(
                insights[category], 
                key=lambda x: list(x.values())[-1], 
                reverse=True
            )[:10]  # Top 10
        
        return insights
    
    def _generate_mood_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate mood analysis based on valence and energy."""
        mood_categories = {
            'happy_energetic': [],
            'happy_calm': [],
            'sad_energetic': [],
            'sad_calm': []
        }
        
        for result in results:
            name = result.get('name', 'Unknown')
            artist = result.get('artist', 'Unknown')
            valence = result.get('valence', 0.5)
            energy = result.get('energy', 0.5)
            
            if valence > 0.6:
                if energy > 0.6:
                    mood_categories['happy_energetic'].append({
                        'name': name,
                        'artist': artist,
                        'valence': valence,
                        'energy': energy
                    })
                else:
                    mood_categories['happy_calm'].append({
                        'name': name,
                        'artist': artist,
                        'valence': valence,
                        'energy': energy
                    })
            else:
                if energy > 0.6:
                    mood_categories['sad_energetic'].append({
                        'name': name,
                        'artist': artist,
                        'valence': valence,
                        'energy': energy
                    })
                else:
                    mood_categories['sad_calm'].append({
                        'name': name,
                        'artist': artist,
                        'valence': valence,
                        'energy': energy
                    })
        
        # Calculate mood distribution
        mood_distribution = {
            category: len(tracks) for category, tracks in mood_categories.items()
        }
        
        return {
            'mood_distribution': mood_distribution,
            'mood_examples': mood_categories
        }
    
    def _calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance-related metrics."""
        metrics = {
            'tempo_ranges': {
                'slow': 0,      # < 90 BPM
                'medium': 0,    # 90-120 BPM
                'fast': 0,      # 120-150 BPM
                'very_fast': 0  # > 150 BPM
            },
            'danceability_levels': {
                'low': 0,       # < 0.4
                'medium': 0,    # 0.4-0.7
                'high': 0       # > 0.7
            }
        }
        
        for result in results:
            tempo = result.get('tempo', 0)
            danceability = result.get('danceability', 0)
            
            # Categorize tempo
            if tempo < 90:
                metrics['tempo_ranges']['slow'] += 1
            elif tempo < 120:
                metrics['tempo_ranges']['medium'] += 1
            elif tempo < 150:
                metrics['tempo_ranges']['fast'] += 1
            else:
                metrics['tempo_ranges']['very_fast'] += 1
            
            # Categorize danceability
            if danceability < 0.4:
                metrics['danceability_levels']['low'] += 1
            elif danceability < 0.7:
                metrics['danceability_levels']['medium'] += 1
            else:
                metrics['danceability_levels']['high'] += 1
        
        return metrics
    
    def _calculate_quality_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality and confidence metrics."""
        confidence_features = [
            'valence_confidence', 'danceability_confidence', 'instrumentalness_confidence',
            'acousticness_confidence', 'speechiness_confidence'
        ]
        
        quality_metrics = {
            'average_confidence': {},
            'high_confidence_tracks': 0,
            'low_confidence_tracks': 0
        }
        
        # Calculate average confidence for each feature
        for feature in confidence_features:
            values = [r.get(feature, 0) for r in results]
            if values:
                quality_metrics['average_confidence'][feature] = round(np.mean(values), 3)
        
        # Count tracks by overall confidence
        for result in results:
            confidences = [result.get(feature, 0) for feature in confidence_features]
            avg_confidence = np.mean(confidences)
            
            if avg_confidence > 0.7:
                quality_metrics['high_confidence_tracks'] += 1
            elif avg_confidence < 0.3:
                quality_metrics['low_confidence_tracks'] += 1
        
        return quality_metrics
    
    def print_summary(self, stats: Dict[str, Any]):
        """Print a human-readable summary of the statistics."""
        print("\n📊 AUDIO ANALYSIS STATISTICS")
        print("=" * 50)
        
        print(f"Total tracks analyzed: {stats.get('total_tracks', 0)}")
        
        # Feature statistics
        feature_stats = stats.get('feature_statistics', {})
        if feature_stats:
            print("\n🎵 FEATURE AVERAGES:")
            for feature, values in feature_stats.items():
                if isinstance(values, dict) and 'mean' in values:
                    print(f"   {feature}: {values['mean']}")
        
        # Mood analysis
        mood_analysis = stats.get('mood_analysis', {})
        mood_dist = mood_analysis.get('mood_distribution', {})
        if mood_dist:
            print("\n😊 MOOD DISTRIBUTION:")
            for mood, count in mood_dist.items():
                print(f"   {mood.replace('_', ' ').title()}: {count}")
        
        # Performance metrics
        perf_metrics = stats.get('performance_metrics', {})
        tempo_ranges = perf_metrics.get('tempo_ranges', {})
        if tempo_ranges:
            print("\n🎼 TEMPO DISTRIBUTION:")
            for range_name, count in tempo_ranges.items():
                print(f"   {range_name.replace('_', ' ').title()}: {count}")
        
        # Quality metrics
        quality_metrics = stats.get('quality_metrics', {})
        if quality_metrics:
            print("\n✅ QUALITY METRICS:")
            high_conf = quality_metrics.get('high_confidence_tracks', 0)
            low_conf = quality_metrics.get('low_confidence_tracks', 0)
            print(f"   High confidence tracks: {high_conf}")
            print(f"   Low confidence tracks: {low_conf}")
        
        print("=" * 50) 
"""
Audio analysis module.
Provides backward compatibility with the original audio_analyzer.py interface.
"""

from .analyzer import AudioAnalyzer

# Backward compatibility - export the main class
__all__ = ['AudioAnalyzer']

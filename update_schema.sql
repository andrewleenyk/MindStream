-- Update tracks table schema to remove unused columns
-- This script removes columns that are not used by the Mozart enhanced analyzer

-- Remove unused columns
ALTER TABLE public.tracks DROP COLUMN IF EXISTS beat_strength;
ALTER TABLE public.tracks DROP COLUMN IF EXISTS rhythmic_stability;
ALTER TABLE public.tracks DROP COLUMN IF EXISTS regularity;

-- Verify the updated schema
-- The table should now have these columns for audio analysis:
-- - tempo, valence, danceability, instrumentalness, acousticness, speechiness
-- - key, mode, energy, loudness
-- - valence_confidence, danceability_confidence, instrumentalness_confidence
-- - acousticness_confidence, speechiness_confidence 
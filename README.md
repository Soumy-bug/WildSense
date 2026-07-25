## Detection Threshold Notes
MegaDetector confidence threshold set at 0.3. Manual review of borderline cases
(e.g., a squirrel detection at 0.43 confidence — visually ambiguous even to a
human reviewer) confirmed this threshold sits in a genuinely uncertain zone
rather than a clean cutoff. This motivates the confidence-based triage feature
planned for the serving layer (Phase 3): low-confidence detections get flagged
for human review rather than auto-logged as confirmed sightings.
import sys
sys.path.insert(0, '/home/youssef/Desktop/vehicles-tracking-id')
errors = []

checks = [
    ("src.geometry.view_geometry", ["compute_cross_view_cosine", "are_views_compatible", "get_shared_visible_stripes"]),
    ("src.matching.cascade_matcher", ["CascadeMatcher", "DetectionInput", "MatchResult", "ActiveTrack"]),
    ("src.matching.hungarian_matcher", ["HungarianMatcher"]),
    ("src.memory.kalman_tracklet", ["KalmanTrackletMemory", "ConstantVelocityKalman"]),
    ("src.memory.cross_view_gallery", ["CrossViewGallery"]),
    ("src.backends.factory", ["build_reid_backend", "build_tracker_backend", "build_gallery_index"]),
]

for module_path, names in checks:
    try:
        mod = __import__(module_path, fromlist=names)
        for name in names:
            getattr(mod, name)
        print(f"  OK  {module_path}")
    except Exception as e:
        print(f"  FAIL  {module_path}: {e}")
        errors.append((module_path, str(e)))

if errors:
    print(f"\n{len(errors)} failures found.")
else:
    print("\nAll new modules import successfully.")

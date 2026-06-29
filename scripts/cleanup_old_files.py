"""
Cleanup script — removes superseded files after the refactoring.
Run once from the project root.
"""
import os

files_to_remove = [
    # Root-level duplicates
    "detector.py",               # replaced by src/detector.py
    "reid.py",                   # absorbed into src/backends/reid/osnet.py
    "test_pipeline.py",          # moved to tests/

    # Legacy src/ files superseded by new packages
    "src/matcher.py",            # replaced by src/matching/hungarian_matcher.py
    "src/matcher_v4.py",         # replaced by src/matching/cascade_matcher.py
    "src/gallery_v4.py",         # replaced by src/memory/cross_view_gallery.py
    "src/tracklet_memory.py",    # replaced by src/memory/kalman_tracklet.py
    "src/view_utils.py",         # replaced by src/geometry/view_geometry.py
    "src/embedder.py",           # replaced by src/feature_extraction/embedder_dispatcher.py

    # Old backend names (kept under new clean names)
    "src/backends/reid/osnet_backend.py",
    "src/backends/reid/transreid_backend.py",
    "src/backends/reid/clipreid_backend.py",
    "src/backends/reid/multibranch_backend.py",
    "src/backends/reid/dinov2_backend.py",
    "src/backends/tracking/strongsort_v2.py",

    # Feature extraction files that moved out of reid/
    "src/backends/reid/view_classifier.py",
    "src/backends/reid/attribute_extractor.py",
    "src/backends/reid/temporal_aggregator.py",

    # GNN matcher that moved to matching/
    "src/backends/matching/gnn_matcher.py",
]

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
removed, missing = [], []

for rel_path in files_to_remove:
    full_path = os.path.join(base, rel_path)
    if os.path.isfile(full_path):
        os.remove(full_path)
        removed.append(rel_path)
    else:
        missing.append(rel_path)

print(f"Removed {len(removed)} files:")
for f in removed:
    print(f"  ✓  {f}")
if missing:
    print(f"\nAlready gone ({len(missing)}):")
    for f in missing:
        print(f"  -  {f}")

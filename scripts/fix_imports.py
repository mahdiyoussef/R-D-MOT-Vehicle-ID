import os, re

base = '/home/youssef/Desktop/vehicles-tracking-id/src'

import_replacements = [
    (r'from src\.view_utils import compute_cross_view_cosine, are_views_compatible',
     'from src.geometry.view_geometry import compute_cross_view_cosine, are_views_compatible'),
    (r'from src\.view_utils import',
     'from src.geometry.view_geometry import'),
    (r'from src\.matcher_v4 import',
     'from src.matching.cascade_matcher import'),
    (r'from src\.gallery_v4 import',
     'from src.memory.cross_view_gallery import'),
    (r'from src\.tracklet_memory import TrackletMemory',
     'from src.memory.kalman_tracklet import KalmanTrackletMemory'),
    (r'from src\.tracklet_memory import',
     'from src.memory.kalman_tracklet import'),
    (r'from src\.backends\.reid\.attribute_extractor import',
     'from src.feature_extraction.attribute_extractor import'),
    (r'from src\.backends\.reid\.view_classifier import',
     'from src.feature_extraction.view_classifier import'),
    (r'from src\.backends\.reid\.temporal_aggregator import',
     'from src.feature_extraction.temporal_aggregator import'),
    (r'from src\.backends\.matching\.gnn_matcher import',
     'from src.matching.gnn_context_matcher import'),
    (r'from src\.backends\.reid\.dinov2_backend import',
     'from src.backends.reid.dinov2_lora import'),
    (r'from src\.backends\.reid\.clipreid_backend import',
     'from src.backends.reid.clip_reid import'),
    (r'from src\.backends\.reid\.multibranch_backend import',
     'from src.backends.reid.multibranch import'),
    (r'from src\.backends\.reid\.transreid_backend import',
     'from src.backends.reid.transreid import'),
    (r'from src\.backends\.reid\.osnet_backend import',
     'from src.backends.reid.osnet import'),
    (r'from src\.backends\.tracking\.strongsort_v2 import',
     'from src.backends.tracking.strongsort import'),
]

# Also fix app.py and factory.py at root level
extra_files = [
    '/home/youssef/Desktop/vehicles-tracking-id/app.py',
    '/home/youssef/Desktop/vehicles-tracking-id/main.py',
    '/home/youssef/Desktop/vehicles-tracking-id/src/backends/factory.py',
    '/home/youssef/Desktop/vehicles-tracking-id/src/pipeline.py',
]

changed = []

def fix_file(fpath):
    content = open(fpath, encoding='utf-8').read()
    new_content = content
    for pattern, replacement in import_replacements:
        new_content = re.sub(pattern, replacement, new_content)
    if new_content != content:
        open(fpath, 'w', encoding='utf-8').write(new_content)
        return True
    return False

# Walk src/
for root, dirs, files in os.walk(base):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for fname in files:
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(root, fname)
        if fix_file(fpath):
            changed.append(os.path.relpath(fpath, base))

# Extra files
for fpath in extra_files:
    if os.path.exists(fpath) and fix_file(fpath):
        changed.append(os.path.relpath(fpath, '/home/youssef/Desktop/vehicles-tracking-id'))

print(f'Updated {len(changed)} files:')
for f in sorted(changed):
    print(f'  {f}')

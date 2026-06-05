# Vehicle Persistent Re-Identification (ReID) Pipeline v2.0

A production-grade, multi-stage computer vision system designed for **persistent vehicle tracking across full occlusion events** (e.g., vehicles passing through tunnels, behind buildings, or entering blind spots).

While standard trackers assign a new ID when a vehicle reappears after a long occlusion, this system extracts deep appearance embeddings and maintains a persistent gallery. When a vehicle re-enters the frame, it is matched against the gallery using the Hungarian Algorithm and Cosine Similarity, successfully recovering its original, global identity.

---

## How the Pipeline Works

### High-Level Data Flow

```
Input Video Frame
       |
       v
+------------------+     +-------------------+     +--------------------+
| Stage 1:         | --> | Stage 2:          | --> | Stage 3:           |
| YOLO Detection   |     | Short-Term Track  |     | Re-ID Embedding    |
| (ONNX Optimized) |     | (BoT-SORT/YOLO)   |     | (TransReID/CLIP)   |
+------------------+     +-------------------+     +--------------------+
                                                           |
                              +----------------------------+
                              v
+------------------+     +-------------------+     +--------------------+
| Stage 6:         | <-- | Stage 5:          | <-- | Stage 4:           |
| Visualization    |     | Hungarian Matcher |     | Persistent Gallery |
| (Annotated Frame)|     | (Optimal Assign.) |     | (EMA + Embeddings) |
+------------------+     +-------------------+     +--------------------+
       |
       v
Annotated Output Frame + JSON Log
```

### Stage-by-Stage Breakdown

#### Stage 1: Vehicle Detection (`src/detector.py`)

- **Model**: YOLOv11m (Ultralytics), exported to **ONNX** for 2-3x inference speedup.
- **Role**: Locates all vehicles in each frame. Returns bounding boxes, confidence scores, and class IDs.
- **ONNX Export**: On first run, the detector automatically exports `.pt` weights to `.onnx` format. Subsequent runs load the cached ONNX model directly.
- **Highway Mode**: Confidence threshold lowered to 0.25 (from 0.4) to detect distant vehicles; NMS IoU set to 0.5 to reduce duplicate boxes.
- **Supported Classes**: car (2), motorcycle (3), bus (5), truck (7) in highway mode; forklift (0) in industrial mode.

#### Stage 2: Short-Term Tracking

Multiple tracker backends are available, selectable via the Streamlit UI:

| Backend | Module | Description |
|---------|--------|-------------|
| **ByteTrack-lite (v1.0)** | `src/tracker.py` | Pure IoU-based association via StrongSORT/boxmot. Fast but no appearance features. |
| **BoT-SORT-ReID** | `src/backends/tracking/botsort_reid.py` | Camera Motion Compensation (CMC) + deep appearance feature fusion + Kalman filtering. Stable under moving cameras. |
| **StrongSORT v2** | `src/backends/tracking/strongsort_v2.py` | Appearance-Free Link (AFLink) + Gaussian process interpolation (GICP) for robust track bridging. |
| **YOLO Native** | `src/backends/tracking/yolo_native.py` | Single-pass detection + tracking using Ultralytics' built-in `.track()` API. Zero-config, lowest latency. |

The tracker assigns short-term track IDs that persist while the vehicle is visible. When a track is lost and re-detected, it gets a new track ID from the tracker -- the Re-ID gallery is responsible for recovering the persistent identity.

#### Stage 3: Appearance Embedding (Re-ID Feature Extraction)

Deep feature vectors are extracted from vehicle crops for identity matching:

| Backend | Module | Embedding Dim | Description |
|---------|--------|:---:|-------------|
| **OSNet** | `src/backends/reid/osnet_backend.py` | 512 | Lightweight CNN backbone (v1.0 legacy). |
| **TransReID** | `src/backends/reid/transreid_backend.py` | 384 | ViT-S/16 backbone with Jigsaw Patch Module (JPM) and Side Information Embeddings (SIE). ICCV 2021. |
| **CLIP-ReID** | `src/backends/reid/clipreid_backend.py` | 512 | CLIP ViT-B/16 with learnable Prompt Learner for text-guided Re-ID. |
| **Multi-Branch** | `src/backends/reid/multibranch_backend.py` | Varies | Global + local part-based features fused via attention or concatenation. |

All embeddings are L2-normalized to ensure consistent cosine similarity scales across backends.

#### Stage 4: Persistent Gallery (`src/gallery.py`)

The core identity memory system. Each gallery entry stores:

```python
{
    "embeddings":        [np.ndarray, ...],   # Rolling window (max 15)
    "ema_representative": np.ndarray,          # Exponential Moving Average embedding
    "color_signature":   np.ndarray,           # HSV color histogram (64-dim)
    "box_size":          (width, height),       # Bounding box dimensions in pixels
    "last_seen":         int,                   # Frame number
    "class":             int,                   # Vehicle class ID
}
```

**Key mechanisms:**
- **EMA Representative**: Instead of simple mean, the gallery maintains an exponentially weighted moving average (`alpha=0.9`) of embeddings for robust long-term representation.
- **Color Histogram Verification**: HSV histograms (H: 32 bins, S: 32 bins) serve as a lightweight secondary filter using Bhattacharyya distance to prevent matching same-model vehicles of different colors.
- **Box-Size Verification** (toggleable): Rejects matches where bounding box dimensions differ beyond a configurable tolerance (default 50%-200% range).
- **Gallery Pruning**: Entries unseen for 5000+ frames are pruned to save memory.

**Gallery Index Backends:**

| Backend | Module | Description |
|---------|--------|-------------|
| **NumPy** | `src/backends/gallery/numpy_index.py` | Brute-force cosine similarity. Simple, exact. |
| **FAISS IVF** | `src/backends/gallery/faiss_ivf_index.py` | Inverted File Index for sub-linear search at scale. |
| **Auto** | `src/backends/gallery/auto_gallery.py` | Starts with NumPy, auto-promotes to FAISS at 1000+ entries. |

#### Stage 5: Identity Matching (`src/matcher.py`)

- **Algorithm**: Hungarian Algorithm (scipy `linear_sum_assignment`)
- **Affinity**: Cosine similarity between query embeddings and gallery representatives
- **Threshold**: Matches below the similarity threshold (default 0.45) are rejected
- **Guarantee**: Optimal one-to-one assignment prevents two queries from claiming the same gallery entry

**Multi-layer identity protection:**
1. Hungarian matcher solves optimal assignment globally
2. `active_pids` set prevents assigning an already-active persistent ID to a new track
3. `batch_assigned_pids` prevents intra-batch duplicate assignments
4. Color histogram verification rejects visually inconsistent matches
5. Box-size verification (optional) rejects dimensionally inconsistent matches

#### Stage 6: Visualization (`src/visualizer.py`)

Renders annotated frames with:
- Color-coded bounding boxes (green=new, orange=recovered, yellow=tracked)
- Persistent ID labels
- Confidence scores
- Trajectory tails (configurable length)

---

## Repository Structure

```text
vehicles-tracking-id/
├── app.py                          # Streamlit web UI
├── main.py                         # CLI entry point
├── configs/
│   └── pipeline.yaml               # Master config: thresholds, model paths, backends
├── models/
│   ├── yolo/
│   │   ├── yolo11m.pt              # YOLO detection weights (PyTorch)
│   │   ├── yolo11m.onnx            # YOLO detection weights (ONNX, auto-exported)
│   │   ├── yolo11n_ha11.pt         # Custom fine-tuned weights
│   │   └── yolo11n_ha11.onnx       # Custom weights (ONNX)
│   └── reid/
│       ├── osnet_veri776.pth        # OSNet checkpoint
│       └── transreid_vit_small_veri776.pth  # TransReID checkpoint
├── src/
│   ├── pipeline.py                 # Main orchestrator (v2.0 strategy pattern)
│   ├── detector.py                 # Stage 1: YOLO detection + ONNX export
│   ├── tracker.py                  # Stage 2: Legacy StrongSORT tracker
│   ├── embedder.py                 # Stage 3: OSNet appearance embedder
│   ├── gallery.py                  # Stage 4: Persistent identity gallery
│   ├── matcher.py                  # Stage 5: Hungarian optimal assignment
│   ├── visualizer.py               # Stage 6: Frame annotation
│   └── backends/
│       ├── factory.py              # Strategy pattern backend factory
│       ├── reid/
│       │   ├── base.py             # Abstract base class
│       │   ├── osnet_backend.py    # OSNet Re-ID backend
│       │   ├── transreid_backend.py # TransReID ViT-S/16 backend
│       │   ├── clipreid_backend.py # CLIP-ReID ViT-B/16 backend
│       │   └── multibranch_backend.py # Global+Local fusion head
│       ├── tracking/
│       │   ├── base.py             # Abstract base class
│       │   ├── legacy_tracker.py   # ByteTrack-lite wrapper
│       │   ├── botsort_reid.py     # BoT-SORT with ReID features
│       │   ├── strongsort_v2.py    # StrongSORT v2 + AFLink
│       │   └── yolo_native.py      # YOLO built-in .track() API
│       └── gallery/
│           ├── base.py             # Abstract base class
│           ├── numpy_index.py      # Brute-force NumPy search
│           ├── faiss_ivf_index.py  # FAISS IVF scalable index
│           └── auto_gallery.py     # Auto-promoting index
├── scripts/
│   ├── train_yolo.py               # YOLO fine-tuning script
│   ├── train_reid.py               # Torchreid training script
│   ├── evaluate_reid.py            # Rank-1/mAP evaluation
│   ├── download_datasets.py        # Kaggle dataset downloader
│   └── convert_*.py                # Dataset format converters
├── tests/                          # Unit tests (63 tests)
└── docs/                           # Reports and documentation
```

---

## Setup & Installation

**1. Create a Python environment:**
```bash
python3 -m venv env
source env/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
pip install onnx onnxslim onnxruntime-gpu   # For ONNX acceleration
```

**3. Launch the Streamlit UI:**
```bash
streamlit run app.py
```

---

## Running Inference

### Via Streamlit UI (Recommended)

```bash
streamlit run app.py
```

The UI allows you to:
- Select pipeline version (v1.0 legacy or v2.0 SOTA)
- Choose scenario mode (Highway or Industrial)
- Configure Re-ID backbone, tracker, and gallery index
- Enable/disable Multi-Branch head and Box-Size filter
- Upload video and watch real-time processing with live metrics

### Via CLI

```bash
python main.py --input /path/to/video.mp4 --output output.mp4
```

**Resume from saved gallery:**
```bash
python main.py --input video.mp4 --resume
```

---

## Configuration (`configs/pipeline.yaml`)

Key tunable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection.confidence_threshold` | 0.4 (0.25 highway) | YOLO confidence threshold |
| `gallery.similarity_threshold` | 0.45 | Min cosine similarity for gallery match |
| `gallery.max_embeddings_per_id` | 15 | Rolling window of embeddings per identity |
| `gallery.gallery_timeout_frames` | 5000 | Frames before pruning stale IDs |
| `matching.threshold` | 0.45 | Hungarian matcher similarity threshold |
| `tracking.max_age` | 60 | Frames to keep lost tracks alive |
| `strategy.reid_backend` | osnet | osnet / transreid / clipreid / multibranch |
| `strategy.tracker_backend` | legacy | legacy / botsort_reid / strongsort_v2 / yolo_native |
| `strategy.gallery_backend` | numpy | numpy / faiss_ivf / auto |

---

## Academic References & SOTA Sources

| Method | Paper | Venue | Repository |
|--------|-------|-------|------------|
| **TransReID** | *Transformer-based Object Re-Identification* | ICCV 2021 | [damo-cv/TransReID](https://github.com/damo-cv/TransReID) |
| **CLIP-ReID** | *CLIP-ReID: Exploiting Reciprocal Relationships for Language-Image Re-ID* | arXiv 2022 | [Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) |
| **BoT-SORT** | *Robust Associations Multi-Pedestrian Tracking* | arXiv 2022 | [NirAharon/BoT-SORT](https://github.com/NirAharon/BoT-SORT) |
| **StrongSORT** | *Make DeepSORT Great Again* | IEEE TCSVT 2023 | [dyhBUPT/StrongSORT](https://github.com/dyhBUPT/StrongSORT) |
| **FAISS** | *Billion-Scale Similarity Search with GPUs* | IEEE TBD 2019 | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| **YOLOv11** | *Ultralytics YOLO11* | Ultralytics 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| **OSNet** | *Omni-Scale Feature Learning for Person Re-ID* | ICCV 2019 | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) |
| **Hungarian Algorithm** | *The Hungarian Method for the Assignment Problem* | Naval Research Logistics, 1955 | scipy `linear_sum_assignment` |

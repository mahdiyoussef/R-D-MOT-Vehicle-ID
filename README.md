# Vehicle Persistent Re-Identification (ReID) Pipeline v4.0 (CROSS-VIEW)

A high-performance, production-grade computer vision system designed for **persistent vehicle tracking across full occlusion events, blind spots, and severe multi-camera viewpoint changes**. 

Standard multi-object trackers (MOT) operate on frame-to-frame association, assigning a new identity when a vehicle is obscured by a bridge, building, or tunnel. This pipeline overcomes this limitation by combining deep feature learning (DINOv2), cross-view geometry awareness, long-term multi-view gallery indexing, and a 5-stage cascade global assignment to persistently maintain vehicle identities.

```text
                  Surveillance / Dashcam Video Input
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 1: VEHICLE DETECTION (YOLOv11m ONNX Accelerated)           │
│ Locates vehicles and outputs bounding boxes, confidence, class   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Bounding Boxes & Crops
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 2: VIEW CLASSIFICATION (MLP Head / Heuristic)              │
│ Tags viewpoint geometry: Front, Rear, Side-L, Side-R, Top-Down   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Viewpoint Tags
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 3: APPEARANCE & SEMANTIC EXTRACTION (DINOv2 + LoRA)        │
│ Extracts L2-norm Global, 6-Stripe Part Embeddings & Attributes   │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Deep Embeddings & OCR/Color
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 4: CROSS-VIEW GALLERY & KINEMATIC MEMORY                   │
│ Maintains view-specific buffers, Kalman predictions & Welford    │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Similarities & Trajectory Priors
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 5: 5-LEVEL CASCADE MATCHING (GATv2 Context + Rules)        │
│ Bipartite assignment: Spatial -> Intra-View -> Inter-View -> Sem │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Resolved Persistent IDs (PIDs)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 6: DYNAMIC VISUALIZATION & LOGGING                         │
│ Renders view-badges, trajectories, and serializes JSON telemetry │
└──────────────────────────────────────────────────────────────────┘
```

---

## 1. System Architecture Deep-Dive

### Stage 1: Vehicle Detection (`src/detector.py`)
Ingests raw frames and outputs vehicle bounding boxes, detection confidences, and class IDs.
*   **Model Backends**: Powered by YOLOv11m. To minimize CPU-GPU transfer bottlenecks, the system automatically compiles `.pt` weights to **ONNX** using `onnxruntime-gpu` on the first execution.
*   **SAHI Support**: Supports Slicing Aided Hyper Inference (SAHI) for ultra-high-resolution panoramic cameras.

### Stage 2: View Classification (`src/backends/reid/view_classifier.py`)
Comparing the embedding of a vehicle's front to its side profile yields a low cosine similarity, causing ID fragmentation. The pipeline prevents this by tagging each detection's geometry before matching.
*   **Architecture**: A lightweight 2-layer MLP head runs on the backbone's CLS token.
*   **Heuristic Fallback**: Aspect-ratio and color symmetry heuristics are applied automatically if custom weights are missing.

### Stage 3: Extraction (DINOv2 + LoRA) (`src/backends/reid/dinov2_lora.py`)
*   **Foundation Model**: Uses DINOv2-ViT-S/14 for zero-shot generalized embeddings.
*   **LoRA Injection**: Injects Low-Rank Adaptation matrices ($r=16$) into the Attention blocks to fine-tune purely on vehicle domains without catastrophic forgetting (trains only 0.8% of weights).
*   **Part-Aware Embeddings**: The image is split into 6 horizontal stripes. Each stripe is encoded independently, allowing the matcher to compare *only* mutually visible parts between different camera angles (e.g., roof is visible from both front and side, but bumpers are not).
*   **Attribute Extractor**: Generates a semantic vector (Color, Class, OCR Plate, Accessories) to serve as an ultimate fallback if visual similarity fails.

### Stage 4: Cross-View Gallery & Tracklet Memory (`src/gallery_v4.py`)
*   **Multi-View Buffers**: Instead of a single average embedding, the gallery stores a circular buffer (length 8) for *each* camera viewpoint (Front, Rear, Side, etc.) per ID.
*   **Kalman Tracklet Memory**: Short-term occlusions (tunnels) are bridged using a frozen Constant Velocity Kalman Filter. If a vehicle reappears near its predicted path, it receives a soft matching bonus (+5% similarity).
*   **Probabilistic Distribution**: The gallery uses Welford's algorithm to model each identity as a Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$, matching via the Mutual Likelihood Score (MLS).

### Stage 5: The 5-Level Cascade Decision Engine (`src/matcher_v4.py`)
1.  **Spatial Poursuit**: Instant match if the incoming bounding box perfectly overlaps the Kalman prediction (IoU > 0.50).
2.  **Intra-View Match**: Strict global embedding comparison if the incoming viewpoint matches the gallery's viewpoint exactly (threshold: 0.75).
3.  **Inter-View Match**: Compares only the mutually visible horizontal stripes between differing viewpoints (threshold: 0.55).
4.  **Semantic Match**: Last resort hybrid score based on Color (35%), Class (25%), Plate (30%), and Accessories (10%).
5.  **New ID Creation**: If all 4 levels fail, a new persistent ID is minted.

---

## 2. Configuration & The Hot-Swap Strategy Pattern

The architecture utilizes a **Strategy Pattern** to decouple the pipeline interface from specific algorithm implementations. The central orchestrator `VehicleReIDPipeline` dynamically instantiates components at startup based on the config file.

### Configuration Schema (`configs/pipeline.yaml`)
```yaml
# configs/pipeline.yaml
strategy:
  reid_backend: "dinov2"         # Options: dinov2 | osnet | transreid | clipreid 
  tracker_backend: "botsort_reid" 
  gallery_backend: "cross_view"  # v4.0 gallery
  use_tensorrt: false            # Toggle dynamic TensorRT compilation

dinov2:
  model_name: "dinov2_vits14"
  lora_rank: 16
  part_stripes: 6
  pretrained_weights: "models/reid/dinov2_lora_veri776.pth"

matcher:
  type: "cascade_v4"
  intra_view_thresh: 0.75
  inter_view_thresh: 0.55
  semantic_thresh: 0.60
```

---

## 3. Developer Extension Guide

### How to Add a Custom Re-ID Backend
1.  Create a new file in `src/backends/reid/custom_reid.py`.
2.  Inherit from `BaseReIDBackend` and implement `load()`, `extract()`, and `extract_batch()`:

```python
# src/backends/reid/custom_reid.py
import numpy as np
from src.backends.reid.base import BaseReIDBackend

class CustomReIDBackend(BaseReIDBackend):
    def __init__(self, config: dict) -> None:
        self.cfg = config.get("custom_reid", {})
        self.device = config["pipeline"]["device"]
        self.model = None

    def load(self) -> None:
        self.model = LoadCustomModel(self.cfg.get("weights")).to(self.device).eval()

    def extract(self, frame: np.ndarray, bbox: np.ndarray, cam_label: int = 0) -> np.ndarray:
        crop = self._preprocess(frame, bbox)
        emb = self.model(crop)
        return emb.cpu().numpy()[0]
```

3.  Register your backend in `src/backends/factory.py` and update `configs/pipeline.yaml`.

---

## 4. Quick Start & CLI Usage

### Installation
```bash
# 1. Create and activate a clean environment
python3 -m venv env
source env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install onnx onnxslim onnxruntime-gpu   # Optional: ONNX speedup
```

### Running the Streamlit Dashboard
```bash
streamlit run app.py
```
This launches the web UI, allowing you to configure the pipeline dynamically, toggle the v4 cascade modules, visualize trajectories, and run side-by-side strategy benchmarks.

### Running via the Command Line
```bash
# Process a video using configs/pipeline.yaml strategies
python main.py --input data/surveillance_highway.mp4 --output outputs/highway_output.mp4

# Run with a saved gallery snapshot to resume vehicle tracking
python main.py --input data/clip2.mp4 --output outputs/clip2_out.mp4 --resume outputs/gallery_snapshot.pkl
```

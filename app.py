import streamlit as st
import os
import json
import cv2
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Vehicle Tracking & ReID", layout="wide")

# ── Page selector ─────────────────────────────────────────────────────────────
PAGE = st.sidebar.radio(
    "📋 Navigation",
    ["🎥 Live Tracking", "⚡ Strategy Benchmark"],
    index=0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Live Tracking (original, UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

if PAGE == "🎥 Live Tracking":

    st.title("Real-Time Vehicle Tracking & Re-Identification")
    st.write("Upload a surveillance or dashcam video to see the AI pipeline process it frame-by-frame in real-time!")

    import torch
    if not torch.cuda.is_available():
        st.warning(
            "⚠️ **NVIDIA GPU Driver Offline / CUDA Not Available**: PyTorch is currently running in CPU mode. "
            "If you have an NVIDIA GPU (e.g., GTX 1650), please verify your NVIDIA drivers and CUDA toolkit are installed. "
            "To prevent runtime crashes, the pipeline has been configured to **automatically fail over to CPU Mode**."
        )

    # --- Sidebar Elements ---
    st.sidebar.header("Real-Time Analytics")
    stats_placeholder = st.sidebar.empty()
    fps_placeholder = st.sidebar.empty()

    st.sidebar.header("Pipeline Settings")

    settings_path = Path("outputs/config/streamlit_settings.json")
    default_settings = {
        "device_choice": "auto",
        "yolo_weights": "outputs/results/weights/best.pt",
        "reid_weights": "models/reid/osnet_x0_25_msmt17.pt",
        "reid_backbone": "vit_base_patch16_224",
        "tracker_config": "botsort.yaml",
        "conf": 0.4,
        "iou": 0.45,
        "imgsz": 640,
        "class_ids": [0, 1],
        "reid_class_ids": [0, 1],
        "similarity_threshold": 0.85,
        "top_k": 5,
        "retention_seconds": 86400,
        "ema_alpha": 0.9,
        "metric": "cosine",
        "update_interval": 5,
        "min_box_area": 5000,
        "frame_interval": 5,
        "milvus_host": "localhost",
        "milvus_port": 19530,
        "milvus_collection": "vehicle_embeddings",
        "milvus_index_type": "HNSW",
    }


    def load_settings() -> dict:
        if settings_path.exists():
            try:
                with open(settings_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {**default_settings, **data}
            except Exception:
                return default_settings.copy()
        return default_settings.copy()


    def save_settings(settings: dict) -> None:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)


    loaded_settings = load_settings()
    for key, value in loaded_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value

    device_choice = st.sidebar.selectbox(
        "Device", ["auto", "cuda:0", "cpu"],
        index=["auto", "cuda:0", "cpu"].index(st.session_state["device_choice"]),
    )
    yolo_weights = st.sidebar.text_input("YOLO Weights", st.session_state["yolo_weights"])
    reid_weights = st.sidebar.text_input("ReID Weights", st.session_state["reid_weights"])
    reid_backbone = st.sidebar.text_input("ReID Backbone", st.session_state["reid_backbone"])
    conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, float(st.session_state["conf"]), 0.05)
    iou = st.sidebar.slider("IoU Threshold", 0.1, 0.9, float(st.session_state["iou"]), 0.05)
    similarity_threshold = st.sidebar.slider(
        "ReID Similarity Threshold", 0.50, 0.99, float(st.session_state["similarity_threshold"]), 0.01
    )
    class_options = [0, 1]
    class_ids = st.sidebar.multiselect(
        "Detection Classes",
        class_options,
        default=st.session_state["class_ids"],
    )
    reid_class_ids = st.sidebar.multiselect(
        "ReID Classes",
        class_options,
        default=st.session_state["reid_class_ids"],
    )

    with st.sidebar.expander("Advanced Settings", expanded=False):
        tracker_config = st.text_input("Tracker Config", st.session_state["tracker_config"])
        imgsz = st.selectbox(
            "Image Size", [320, 480, 640, 960],
            index=[320, 480, 640, 960].index(int(st.session_state["imgsz"]))
        )
        ema_alpha = st.slider("EMA Alpha", 0.50, 0.99, float(st.session_state["ema_alpha"]), 0.01)
        metric = st.selectbox("DB Metric", ["cosine", "l2"], index=["cosine", "l2"].index(st.session_state["metric"]))
        update_interval = st.slider("ReID Update Interval (frames)", 1, 30, int(st.session_state["update_interval"]), 1)
        min_box_area = st.slider("Min Box Area", 0, 20000, int(st.session_state["min_box_area"]), 250)
        frame_interval = st.slider("Extraction Frame Interval", 1, 30, int(st.session_state["frame_interval"]), 1)
        top_k = st.slider("Top-K Search", 1, 20, int(st.session_state["top_k"]), 1)
        retention_seconds = st.slider(
            "Retention Seconds",
            0,
            172800,
            int(st.session_state["retention_seconds"]),
            600,
        )
        milvus_host = st.text_input("Milvus Host", st.session_state["milvus_host"])
        milvus_port = st.number_input("Milvus Port", min_value=1, max_value=65535, value=int(st.session_state["milvus_port"]))
        milvus_collection = st.text_input("Milvus Collection", st.session_state["milvus_collection"])
        milvus_index_type = st.text_input("Milvus Index Type", st.session_state["milvus_index_type"])

    current_settings = {
        "device_choice": device_choice,
        "yolo_weights": yolo_weights,
        "reid_weights": reid_weights,
        "reid_backbone": reid_backbone,
        "tracker_config": tracker_config,
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "class_ids": class_ids,
        "reid_class_ids": reid_class_ids,
        "similarity_threshold": float(similarity_threshold),
        "top_k": int(top_k),
        "retention_seconds": int(retention_seconds),
        "ema_alpha": float(ema_alpha),
        "metric": metric,
        "update_interval": int(update_interval),
        "min_box_area": int(min_box_area),
        "frame_interval": int(frame_interval),
        "milvus_host": milvus_host,
        "milvus_port": int(milvus_port),
        "milvus_collection": milvus_collection,
        "milvus_index_type": milvus_index_type,
    }
    save_settings(current_settings)

    st.markdown("---")
    st.subheader("🤖 Pipeline Architecture & Strategy Selector")
    st.write("Select the underlying pipeline version and hot-swap advanced AI algorithms in real-time.")

    pipeline_choice = st.radio(
        "Select Pipeline Architecture",
        [
            "v3.0 ULTRA Vehicle Re-ID Pipeline (GNN + DINOv2)",
            "v2.0 SOTA Vehicle Re-ID Pipeline (Multi-Strategy)", 
            "v1.0 Legacy Forklift Tracking Pipeline (Milvus-Backed)"
        ],
        index=0,
        horizontal=True,
    )

    scenario_choice = st.radio(
        "🚗 Target Scenario Mode",
        ["Industrial Mode (Detect Forklifts & Warehouse Staff)", "Highway Mode (Detect Cars, Trucks, Buses, Motorcycles)"],
        index=0,
        horizontal=True,
    )

    # Initialize variables with defaults
    reid_choice = "TransReID (ViT-Base/16) [SOTA]"
    tracker_choice = "StrongSORT v2 (AFLink+GICP) [SOTA]"
    gallery_choice = "Auto-Upgrade Index [Recommended]"
    multibranch_enabled = True

    if pipeline_choice in ["v2.0 SOTA Vehicle Re-ID Pipeline (Multi-Strategy)", "v3.0 ULTRA Vehicle Re-ID Pipeline (GNN + DINOv2)"]:
        is_v3 = "v3.0" in pipeline_choice
        
        st.markdown(
            f"<div style='border: 1px solid #2196f3; border-radius: 8px; padding: 15px; margin-bottom: 20px; background-color: rgba(33, 150, 243, 0.05);'>"
            f"<h4 style='color: #2196f3; margin-top: 0;'>🚀 {'v3.0 ULTRA' if is_v3 else 'v2.0 SOTA'} Hot-Swap Strategy Control Panel</h4>"
            f"Enable or disable SOTA tracking, Re-ID backbones, custom heads, and database indexes dynamically below."
            "</div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            reid_choice = st.selectbox(
                "🧬 Re-ID Backbone Strategy",
                ["DINOv2 (ViT-Small/14) + LoRA [v3.0]", "TransReID (ViT-Base/16) [SOTA]", "CLIP-ReID (ViT-B/16) [SOTA]", "OSNet (CNN) [v1.0 Legacy]"],
                index=0 if is_v3 else 1,
            )
            with st.popover("❓ Learn about Re-ID Backbones", width="stretch"):
                st.markdown("### 🧬 Re-ID Neural Backbones")
                st.graphviz_chart('''
                    digraph G {
                        rankdir=LR;
                        node [shape=box, style="rounded,filled", fillcolor="#f0f8ff", fontname="Arial", fontsize=11];
                        edge [fontname="Arial", fontsize=10];
                        Image [shape=cylinder, fillcolor="#ffd700", label="Vehicle Crop"];
                        OSNet [label="OSNet\\n(CNN, Fast\\nCPU/GPU)"];
                        TransReID [label="TransReID\\n(ViT, Robust\\nJPM+SIE)"];
                        CLIP [label="CLIP-ReID\\n(Zero-Shot\\nVision-Language)"];
                        DINOv2 [label="DINOv2+LoRA\\n(v3.0 SOTA\\nSelf-Supervised)"];
                        Embedding [shape=ellipse, fillcolor="#98fb98", label="Visual\\nSignature"];
                        
                        Image -> OSNet;
                        Image -> TransReID;
                        Image -> CLIP;
                        Image -> DINOv2;
                        
                        OSNet -> Embedding;
                        TransReID -> Embedding;
                        CLIP -> Embedding;
                        DINOv2 -> Embedding;
                    }
                ''')

            gallery_choice = st.selectbox(
                "🗂️ Gallery Search Index",
                ["Probabilistic Gallery (Gaussian μ+σ) [v3.0]", "Auto-Upgrade Index [Recommended]", "FAISS IVFFlat Index", "Flat Numpy Index [v1.0 Legacy]"],
                index=0 if is_v3 else 1,
            )
            with st.popover("❓ Learn about Gallery Indexes", width="stretch"):
                st.markdown("### 🗂️ Gallery Search Indexes")
                st.graphviz_chart('''
                    digraph G {
                        rankdir=TB;
                        node [shape=box, style="rounded,filled", fillcolor="#ffe4e1", fontname="Arial", fontsize=11];
                        edge [fontname="Arial", fontsize=10];
                        Start [shape=point];
                        Numpy [label="Flat Numpy\\n(Small scale, Exact Cosine)"];
                        FAISS [label="FAISS IVFFlat\\n(Large scale, K-Means)"];
                        Auto [label="Auto-Upgrade Index\\n(Dynamic Switching)", fillcolor="#e6e6fa"];
                        Prob [label="Probabilistic Gallery\\n(v3.0 Gaussian μ+σ, MLS)", fillcolor="#ffd700"];
                        
                        Start -> Numpy [label="  Basic Option"];
                        Start -> FAISS [label="  Scale Option"];
                        Start -> Auto [label="  Smart Option"];
                        Start -> Prob [label="  SOTA Option"];
                        
                        Auto -> Numpy [label="< 1000 IDs"];
                        Auto -> FAISS [label="> 1000 IDs"];
                    }
                ''')

        with col2:
            tracker_choice = st.selectbox(
                "🏃 Short-Term Motion Tracker",
                [
                    "BoT-SORT-ReID (CMC+Kalman) [SOTA]",
                    "StrongSORT v2 (AFLink+GICP) [SOTA]",
                    "YOLO Native Tracker (Built-in BoT-SORT/ByteTrack)",
                    "Legacy ByteTrack-lite [v1.0 Legacy]",
                ],
                index=0 if is_v3 else 1,
            )
            with st.popover("❓ Learn about Motion Trackers", width="stretch"):
                st.markdown("### 🏃 Short-Term Motion Trackers")
                st.graphviz_chart('''
                    digraph G {
                        rankdir=LR;
                        node [shape=box, style="rounded,filled", fillcolor="#e6e6fa", fontname="Arial", fontsize=11];
                        edge [fontname="Arial", fontsize=10];
                        YOLO [label="YOLO Detections", fillcolor="#ffd700"];
                        ByteTrack [label="ByteTrack-lite\\n(v1.0 IoU only)"];
                        Kalman [label="BoT-SORT-ReID\\n(+CMC & Kalman)"];
                        StrongSORT [label="StrongSORT v2\\n(+AFLink & GICP)"];
                        Native [label="YOLO Native\\n(Built-in single-pass)"];
                        
                        YOLO -> ByteTrack;
                        YOLO -> Native;
                        YOLO -> Kalman [label=" Adds Motion\\nComp (CMC)"];
                        Kalman -> StrongSORT [label=" Adds Track\\nSmoothing"];
                    }
                ''')

            multibranch_enabled = st.toggle("✨ Enable Multi-Branch Global+Local Head", value=True)
            with st.popover("❓ Learn about Multi-Branch Head", width="stretch"):
                st.markdown(
                    "### ✨ Multi-Branch Global+Local Head\n"
                    "Extracts part-level spatial details alongside global vehicle shapes to form high-fidelity signatures.\n\n"
                    "- **Enabled**: Vehicle crops are divided into horizontal stripes (local branches) and combined with global features via attention or concatenation. Dramatic boost in accuracy under structural occlusions.\n"
                    "- **Disabled**: Falls back to simple global feature representation."
                )

            tensorrt_enabled = st.toggle("🚀 Enable TensorRT (TRT) Acceleration for Re-ID", value=False)
            with st.popover("❓ Learn about TensorRT", width="stretch"):
                st.markdown(
                    "### 🚀 TensorRT Acceleration\n"
                    "Compiles the PyTorch Re-ID backbone to an NVIDIA TensorRT engine using `torch_tensorrt`.\n\n"
                    "- **Enabled**: Model is traced and compiled to TRT on first load. Inference can be 2-4x faster. Note: First load may take a few minutes.\n"
                    "- **Disabled**: Standard PyTorch execution."
                )

            box_size_filter_enabled = st.toggle(
                "📏 Enable Box-Size Consistency Filter",
                value=False,
                help="When enabled, the gallery will reject Re-ID matches if the "
                     "bounding box dimensions differ too much from the stored entry. "
                     "Useful to prevent matching a distant small vehicle to a nearby large one.",
            )
            if box_size_filter_enabled:
                box_size_tolerance = st.slider(
                    "Box-Size Tolerance",
                    min_value=0.2,
                    max_value=0.9,
                    value=0.5,
                    step=0.05,
                    help="Minimum dimension ratio (width & height independently). "
                         "0.5 means each dimension must be within 50%-200% of stored size. "
                         "Lower = stricter, Higher = more permissive.",
                )
            else:
                box_size_tolerance = 0.5
                
        if is_v3:
            st.markdown("#### ✨ v3.0 Advanced Modules")
            col3, col4 = st.columns(2)
            with col3:
                gnn_matcher_enabled = st.toggle("🧠 Enable GNN Context-Aware Matcher [v3.0]", value=True)
                with st.popover("❓ Learn about GNN Matcher", width="stretch"):
                    st.markdown("### 🧠 GNN Context-Aware Matcher")
                    st.graphviz_chart('''
                        digraph G {
                            rankdir=LR;
                            node [shape=box, style="rounded,filled", fillcolor="#f5f5dc", fontname="Arial", fontsize=11];
                            Detections [label="Vehicle\\nDetections"];
                            Gallery [label="Gallery\\nIdentities"];
                            GATv2 [label="Graph Attention\\nNetwork (GATv2)", fillcolor="#add8e6"];
                            Hungarian [label="Hungarian\\nAssignment", fillcolor="#98fb98"];
                            Matches [shape=ellipse, label="Robust\\nAssociations"];
                            
                            Detections -> GATv2 [label=" Nodes"];
                            Gallery -> GATv2 [label=" Nodes"];
                            GATv2 -> Hungarian [label=" Context-Aware\\nAffinity Matrix"];
                            Hungarian -> Matches;
                        }
                    ''')
            with col4:
                temporal_aggregator_enabled = st.toggle("⏱️ Enable Temporal Tracklet Aggregator [v3.0]", value=True)
                with st.popover("❓ Learn about Temporal Aggregator", width="stretch"):
                    st.markdown("### ⏱️ Temporal Tracklet Aggregator")
                    st.graphviz_chart('''
                        digraph G {
                            rankdir=LR;
                            node [shape=box, style="rounded,filled", fillcolor="#ffe4b5", fontname="Arial", fontsize=11];
                            F1 [label="Frame t-2\\n(Clear)"];
                            F2 [label="Frame t-1\\n(Blurry)"];
                            F3 [label="Frame t\\n(Clear)"];
                            Attention [label="Cross-Attention\\nTransformer", fillcolor="#ffb6c1"];
                            Fused [shape=ellipse, fillcolor="#98fb98", label="Robust Fused\\nEmbedding"];
                            
                            F1 -> Attention [label=" High Weight"];
                            F2 -> Attention [label=" Low Weight"];
                            F3 -> Attention [label=" High Weight"];
                            Attention -> Fused;
                        }
                    ''')

    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        if st.button("Start Real-Time Processing", type="primary"):
            # 1. Save uploaded file to temp directory
            os.makedirs("outputs/temp", exist_ok=True)
            input_path = f"outputs/temp/{uploaded_file.name}"
            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())
                
            # 2. Setup video capture
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Error opening video stream.")
                st.stop()
                
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fps_placeholder.write(f"**Source Video FPS:** {video_fps:.2f}")
            
            # 3. Import and initialize the pipeline natively inside the app
            with st.spinner("Loading AI Models & Pipeline Backends..."):
                device = None if device_choice == "auto" else device_choice
                if not torch.cuda.is_available():
                    device = "cpu"
                
                if pipeline_choice in ["v2.0 SOTA Vehicle Re-ID Pipeline (Multi-Strategy)", "v3.0 ULTRA Vehicle Re-ID Pipeline (GNN + DINOv2)"]:
                    import yaml
                    import torch
                    from src.pipeline import VehicleReIDPipeline
                    
                    # Load template config
                    with open("configs/pipeline.yaml") as f:
                        cfg = yaml.safe_load(f)
                    
                    # Set active device
                    active_device = "cuda:0" if device_choice == "auto" and torch.cuda.is_available() else device_choice
                    if not torch.cuda.is_available():
                        active_device = "cpu"
                    cfg["pipeline"]["device"] = active_device
                    
                    if active_device == "cpu":
                        cfg["pipeline"]["half_precision"] = False
                        cfg["tracking"]["half"] = False
                    
                    # Apply scenario-specific settings
                    if scenario_choice == "Highway Mode (Detect Cars, Trucks, Buses, Motorcycles)":
                        cfg["detection"]["weights"] = "models/yolo/yolo11m.onnx"
                        cfg["detection"]["fallback_weights"] = "yolo11m.pt"
                        cfg["detection"]["confidence_threshold"] = 0.25  # lower for distant highway vehicles
                        cfg["detection"]["iou_threshold"] = 0.5
                        cfg["detection"]["reid_class_ids"] = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
                        cfg["detection"]["vehicle_classes"] = {
                            2: "car",
                            3: "motorcycle",
                            5: "bus",
                            7: "truck",
                        }
                    else:
                        cfg["detection"]["weights"] = "models/yolo/yolo11n_ha11.onnx"
                        cfg["detection"]["fallback_weights"] = "yolo11m.pt"
                        cfg["detection"]["reid_class_ids"] = [0]  # Forklift only
                        cfg["detection"]["vehicle_classes"] = {
                            0: "forklift",
                            1: "person",
                        }
                    
                    # Map Re-ID Backbone
                    if "DINOv2" in reid_choice:
                        reid_backend = "dinov2"
                    elif "TransReID" in reid_choice:
                        reid_backend = "transreid"
                    elif "CLIP-ReID" in reid_choice:
                        reid_backend = "clipreid"
                    else:
                        reid_backend = "osnet"
                    
                    # Map Short-Term Tracker
                    if "StrongSORT v2" in tracker_choice:
                        tracker_backend = "strongsort"
                    elif "BoT-SORT-ReID" in tracker_choice:
                        tracker_backend = "botsort_reid"
                    elif "YOLO Native Tracker" in tracker_choice:
                        tracker_backend = "yolo_native"
                    else:
                        tracker_backend = "legacy"
                    
                    # Map Gallery Index
                    if "Probabilistic" in gallery_choice:
                        gallery_backend = "probabilistic"
                    elif "Auto-Upgrade" in gallery_choice:
                        gallery_backend = "auto"
                    elif "FAISS" in gallery_choice:
                        gallery_backend = "faiss_ivf"
                    else:
                        gallery_backend = "numpy"
                        
                    matcher_backend = "gnn" if locals().get("gnn_matcher_enabled", True) else "hungarian"
                    
                    # Setup Multi-Branch Head
                    if multibranch_enabled and reid_backend != "dinov2":
                        cfg["strategy"] = {
                            "reid_backend": "multibranch",
                            "tracker_backend": tracker_backend,
                            "gallery_backend": gallery_backend,
                            "matcher_backend": matcher_backend,
                            "use_tensorrt": tensorrt_enabled
                        }
                        cfg["multibranch"]["backbone"] = reid_backend
                    else:
                        cfg["strategy"] = {
                            "reid_backend": reid_backend,
                            "tracker_backend": tracker_backend,
                            "gallery_backend": gallery_backend,
                            "matcher_backend": matcher_backend,
                            "use_tensorrt": tensorrt_enabled
                        }
                    
                    # Setup v3.0 Advanced Modules
                    cfg["temporal_aggregator"]["enabled"] = locals().get("temporal_aggregator_enabled", True)
                    
                    # Inject box-size filter settings into gallery config
                    cfg["gallery"]["box_size_filter_enabled"] = box_size_filter_enabled
                    cfg["gallery"]["box_size_tolerance"] = box_size_tolerance

                    # Save active configuration
                    os.makedirs("outputs/config", exist_ok=True)
                    active_config_path = "outputs/config/streamlit_active_v2.yaml"
                    with open(active_config_path, "w") as f:
                        yaml.dump(cfg, f)
                    
                    pipeline = VehicleReIDPipeline(config_path=active_config_path)

                    # Apply box-size filter settings to the gallery
                    pipeline.gallery.box_size_filter_enabled = box_size_filter_enabled
                    pipeline.gallery.box_size_tolerance = box_size_tolerance
                    
                else:
                    from main import ForkliftTrackingPipeline
                    pipeline = ForkliftTrackingPipeline(
                        yolo_weights=yolo_weights,
                        reid_weights=reid_weights,
                        reid_backbone=reid_backbone,
                        tracker_config=tracker_config,
                        device=device,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        class_ids=class_ids,
                        reid_class_ids=reid_class_ids,
                        similarity_threshold=similarity_threshold,
                        top_k=top_k,
                        retention_seconds=retention_seconds,
                        ema_alpha=ema_alpha,
                        metric=metric,
                        update_interval=update_interval,
                        min_box_area=min_box_area,
                        frame_interval=frame_interval,
                        milvus_host=milvus_host,
                        milvus_port=milvus_port,
                        milvus_collection=milvus_collection,
                        milvus_index_type=milvus_index_type,
                    )
            
            # 4. Setup placeholders for real-time streaming
            st.subheader("Live Detection Stream")
            video_placeholder = st.empty()
            
            # Real-time state tracking
            vehicle_stats = {}  # pid -> frame count
            
            # 5. Process loop
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the exact frame through the 6-stage pipeline
                annotated_frame, track_records = pipeline.process_frame(frame)
                
                # Update analytics
                for rec in track_records:
                    pid = rec.get("persistent_id", -1)
                    if pid != -1:
                        vehicle_stats[pid] = vehicle_stats.get(pid, 0) + 1
                
                # Draw real-time sidebar dataframe
                if vehicle_stats:
                    df = pd.DataFrame(list(vehicle_stats.items()), columns=["Vehicle ID", "Total Frames Detected"])
                    df["Duration (Seconds)"] = (df["Total Frames Detected"] / video_fps).round(2)
                    
                    # Sort by duration so the longest-seen vehicles stay at the top
                    df = df.sort_values(by="Duration (Seconds)", ascending=False)
                    
                    stats_placeholder.dataframe(
                        df,
                        hide_index=True,
                        width='stretch'
                    )
                
                # Convert BGR (OpenCV) to RGB (Streamlit)
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame instantly
                video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                
            cap.release()
            st.success("Real-Time Processing Complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — ⚡ Strategy Benchmark (NEW in v2.0)
# ─────────────────────────────────────────────────────────────────────────────

elif PAGE == "⚡ Strategy Benchmark":
    import time
    import yaml

    st.title("⚡ Strategy Benchmark")
    st.markdown(
        """
        Upload a video clip (max 60 s recommended) and compare all three pipeline
        presets **side-by-side** on identical input.

        | Preset | Re-ID | Tracker | Gallery |
        |--------|-------|---------|---------|
        | **FAST**     | OSNet | Legacy StrongSORT | Numpy |
        | **BALANCED** | TransReID ViT-S | BoT-SORT-ReID | FAISS auto |
        | **SOTA**     | CLIP-ReID ViT-B | StrongSORT v2 | FAISS IVF |
        """
    )

    # ── Upload + settings ─────────────────────────────────────────────────────
    bm_col1, bm_col2 = st.columns([3, 1])
    with bm_col1:
        bm_video = st.file_uploader(
            "📁 Upload Video Clip", type=["mp4", "avi", "mov"], key="bm_upload"
        )
    with bm_col2:
        max_frames = st.number_input("⚙ Max Frames", min_value=10, max_value=2000, value=100, step=10)

    run_btn = st.button("▶ Run All Presets", type="primary", disabled=(bm_video is None))

    # ── Preset definitions ────────────────────────────────────────────────────
    PRESETS = [
        {
            "name": "FAST (v1.0)",
            "config_path": "configs/preset_fast.yaml",
            "reid":    "OSNet",
            "tracker": "Legacy StrongSORT",
            "gallery": "Numpy",
            "color":   "#4caf50",
        },
        {
            "name": "BALANCED",
            "config_path": "configs/preset_balanced.yaml",
            "reid":    "TransReID ViT-S",
            "tracker": "BoT-SORT-ReID",
            "gallery": "FAISS auto",
            "color":   "#2196f3",
        },
        {
            "name": "SOTA",
            "config_path": "configs/preset_sota.yaml",
            "reid":    "CLIP-ReID ViT-B",
            "tracker": "StrongSORT v2",
            "gallery": "FAISS IVF",
            "color":   "#9c27b0",
        },
    ]

    def _run_preset(config_path: str, video_path: str, max_frames: int) -> dict:
        """
        Run a single pipeline preset on the video and collect per-frame metrics.
        Returns a summary dict with fps, active_ids, reid_events, idsw, sample_frames.
        """
        import yaml, time, cv2, numpy as np
        from src.pipeline import VehicleReIDPipeline

        with open(config_path) as f:
            full_cfg = yaml.safe_load(f)

        pipeline = VehicleReIDPipeline(config_path=config_path)

        cap = cv2.VideoCapture(video_path)
        frame_metrics = []
        sample_frames = []
        id_history    = {}   # pid → set of tracker_ids (for IDSW counting)
        reid_events   = 0
        idsw_count    = 0
        prev_pid_map  = {}
        t0            = time.perf_counter()

        frame_n = 0
        while cap.isOpened() and frame_n < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, records = pipeline.process_frame(frame)

            # Detect Re-ID events (status == 'recovered')
            n_reid = sum(1 for r in records if r.get("status") == "recovered")
            reid_events += n_reid

            # Detect identity switches: same tracker_id now has different pid
            for r in records:
                tid = r.get("track_id", -1)
                pid = r.get("persistent_id", -1)
                if tid in prev_pid_map and prev_pid_map[tid] != pid and pid != -1:
                    idsw_count += 1
                prev_pid_map[tid] = pid

            elapsed = time.perf_counter() - t0
            fps_now = (frame_n + 1) / max(elapsed, 1e-6)

            frame_metrics.append({
                "frame": frame_n,
                "fps":   round(fps_now, 2),
                "active": len(set(r["persistent_id"] for r in records if r["persistent_id"] != -1)),
                "gallery_size": len(pipeline.gallery),
                "reid_events": reid_events,
                "idsw": idsw_count,
                "gallery_promoted": getattr(
                    getattr(pipeline, "_gallery_index", None), "_promoted", False
                ),
            })

            # Capture 5 evenly-spaced sample frames
            if frame_n % max(1, max_frames // 5) == 0 and len(sample_frames) < 5:
                sample_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            frame_n += 1

        cap.release()
        total_elapsed = time.perf_counter() - t0

        return {
            "fps_avg":      round(frame_n / max(total_elapsed, 1e-6), 1),
            "active_ids":   frame_metrics[-1]["active"]   if frame_metrics else 0,
            "gallery_size": frame_metrics[-1]["gallery_size"] if frame_metrics else 0,
            "reid_events":  reid_events,
            "idsw":         idsw_count,
            "frame_metrics": frame_metrics,
            "sample_frames": sample_frames,
            "promoted_at": next(
                (m["frame"] for m in frame_metrics if m.get("gallery_promoted")),
                None,
            ),
        }

    # ── Run benchmarks ────────────────────────────────────────────────────────
    if run_btn and bm_video is not None:
        os.makedirs("outputs/temp", exist_ok=True)
        video_path = f"outputs/temp/benchmark_{bm_video.name}"
        with open(video_path, "wb") as f:
            f.write(bm_video.read())

        if "bm_results" not in st.session_state:
            st.session_state["bm_results"] = {}

        results = {}
        for preset in PRESETS:
            status_box = st.status(f"Running **{preset['name']}**…", expanded=True)
            progress   = st.progress(0)
            try:
                with status_box:
                    st.write(f"Re-ID: {preset['reid']} | Tracker: {preset['tracker']} | Gallery: {preset['gallery']}")
                res = _run_preset(preset["config_path"], video_path, int(max_frames))
                results[preset["name"]] = res
                status_box.update(label=f"✅ {preset['name']} done ({res['fps_avg']} FPS)", state="complete")
            except Exception as e:
                results[preset["name"]] = {"error": str(e)}
                status_box.update(label=f"❌ {preset['name']} failed: {e}", state="error")
            progress.progress(1.0)

        st.session_state["bm_results"] = results

    # ── Display results ───────────────────────────────────────────────────────
    if "bm_results" in st.session_state and st.session_state["bm_results"]:
        results = st.session_state["bm_results"]

        st.markdown("---")
        st.subheader("📊 Side-by-Side Comparison")

        col_fast, col_bal, col_sota = st.columns(3)
        cols = [col_fast, col_bal, col_sota]

        for col, preset in zip(cols, PRESETS):
            name = preset["name"]
            res  = results.get(name, {})

            with col:
                st.markdown(
                    f"<h4 style='color:{preset['color']};'>{name}</h4>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Re-ID: **{preset['reid']}**  \nTracker: **{preset['tracker']}**  \nGallery: **{preset['gallery']}**")

                if "error" in res:
                    st.error(f"Error: {res['error']}")
                    continue

                # Metrics table
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| **FPS** | {res.get('fps_avg', '—')} |
| **Active Tracks** | {res.get('active_ids', '—')} |
| **Gallery Size** | {res.get('gallery_size', '—')} |
| **Re-IDs** | {res.get('reid_events', '—')} |
| **IDSW** | {res.get('idsw', '—')} |
""")
                # Sample frames
                frames = res.get("sample_frames", [])
                if frames:
                    st.image(frames, use_container_width=True, caption=[f"Frame {i}" for i in range(len(frames))])

        # ── Charts ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📈 Comparison Charts")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "FPS Over Time", "Re-ID Events Over Time",
                    "Gallery Growth", "Identity Switches Cumulative",
                ],
            )

            for i, preset in enumerate(PRESETS):
                name = preset["name"]
                res  = results.get(name, {})
                if "error" in res or "frame_metrics" not in res:
                    continue
                fm    = res["frame_metrics"]
                x     = [m["frame"] for m in fm]
                color = preset["color"]

                # FPS
                fig.add_trace(go.Scatter(x=x, y=[m["fps"] for m in fm],
                    name=name, line=dict(color=color), legendgroup=name,
                    showlegend=(i == 0)), row=1, col=1)

                # Re-ID events
                fig.add_trace(go.Scatter(x=x, y=[m["reid_events"] for m in fm],
                    name=name, line=dict(color=color), legendgroup=name,
                    showlegend=False), row=1, col=2)

                # Gallery growth
                gallery_trace = go.Scatter(x=x, y=[m["gallery_size"] for m in fm],
                    name=name, line=dict(color=color), legendgroup=name, showlegend=False)
                fig.add_trace(gallery_trace, row=2, col=1)

                # Annotate FAISS promotion point
                promoted_at = res.get("promoted_at")
                if promoted_at is not None:
                    gallery_size_at = next(
                        (m["gallery_size"] for m in fm if m["frame"] == promoted_at), 0
                    )
                    fig.add_vline(
                        x=promoted_at, line_dash="dash", line_color=color,
                        annotation_text=f"FAISS↑", row=2, col=1,
                    )

                # IDSW cumulative
                fig.add_trace(go.Scatter(x=x, y=[m["idsw"] for m in fm],
                    name=name, line=dict(color=color), legendgroup=name,
                    showlegend=False), row=2, col=2)

            fig.update_layout(height=600, title_text="Strategy Benchmark — Metrics Timeline")
            st.plotly_chart(fig, width="stretch")

        except ImportError:
            st.info("Install plotly for charts: `pip install plotly`")

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("---")
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            report = {
                name: {k: v for k, v in res.items() if k != "sample_frames"}
                for name, res in results.items()
            }
            st.download_button(
                "💾 Download Comparison Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name="benchmark_report.json",
                mime="application/json",
            )

        with export_col2:
            summary_rows = []
            for preset in PRESETS:
                name = preset["name"]
                res  = results.get(name, {})
                if "error" not in res:
                    summary_rows.append({
                        "Preset": name, "ReID": preset["reid"],
                        "Tracker": preset["tracker"], "Gallery": preset["gallery"],
                        "FPS": res.get("fps_avg", "—"),
                        "Re-IDs": res.get("reid_events", "—"),
                        "IDSW": res.get("idsw", "—"),
                    })
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")


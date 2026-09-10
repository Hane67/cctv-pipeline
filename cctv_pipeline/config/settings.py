"""
Pydantic configuration models for the CCTV Video Analytics Pipeline.
Provides strong type validation, defaults, and serialization to/from YAML/JSON.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Configuration for Stage 1: Adaptive Illumination & Artifact Preprocessing."""
    enable_clahe: bool = Field(True, description="Enable CLAHE illumination normalization")
    clahe_clip_limit: float = Field(2.5, description="CLAHE contrast limiting factor")
    clahe_tile_grid: Tuple[int, int] = Field((8, 8), description="Grid size for CLAHE tiles")

    enable_gamma: bool = Field(True, description="Enable gamma correction")
    auto_gamma: bool = Field(True, description="Dynamically determine gamma based on scene mean luminance")
    gamma_value: float = Field(1.4, description="Fallback/fixed gamma value if auto_gamma is False")

    enable_deband: bool = Field(True, description="Enable selective gradient-based debanding (AdaDeband-style)")
    deband_flat_thresh: float = Field(8.0, description="Gradient threshold separating banding from edges")
    deband_blur_ksize: int = Field(9, description="Kernel size for selective debanding blur")

    enable_deblock: bool = Field(True, description="Enable edge-preserving deblocking (VRCNN bilateral stand-in)")
    deblock_d: int = Field(5, description="Diameter of each pixel neighborhood for deblocking")
    deblock_sigma_color: float = Field(50.0, description="Deblock filter sigma in color space")
    deblock_sigma_space: float = Field(50.0, description="Deblock filter sigma in coordinate space")

    enable_denoise: bool = Field(False, description="Enable spatial/temporal denoising (recommended for low-light/night)")
    auto_night_denoise: bool = Field(True, description="Automatically enable denoising when scene is classified as dark/night")
    denoise_h: float = Field(7.0, description="Filter strength for luminance component")
    denoise_h_color: float = Field(7.0, description="Filter strength for color components")


class SegmentationConfig(BaseModel):
    """Configuration for Stage 2: Illumination-Invariant Foreground Segmentation."""
    algorithm: str = Field("subsense", description="Foreground segmentation method: 'subsense', 'mog2', 'knn'")
    n_samples: int = Field(20, description="Number of background samples per pixel in SuBSENSE buffer")
    lbsp_thresh: int = Field(8, description="Relative contrast threshold for 8-bit Local Binary Similarity Patterns")
    init_dist_thresh: float = Field(24.0, description="Initial distance threshold R(x)")
    min_dist_thresh: float = Field(16.0, description="Lower bound for per-pixel dynamic threshold R(x)")
    max_dist_thresh: float = Field(80.0, description="Upper bound for per-pixel dynamic threshold R(x)")
    match_count_req: int = Field(2, description="Minimum matching samples required to classify as background")
    intensity_weight: float = Field(0.05, description="Minor weight for raw intensity difference vs LBSP Hamming distance")

    morph_open_ksize: int = Field(3, description="Kernel size for morphological opening (speckle removal)")
    morph_close_ksize: int = Field(7, description="Kernel size for morphological closing (hole filling)")

    # Baseline MOG2 / KNN parameters
    mog2_history: int = Field(500, description="MOG2 history frames")
    mog2_var_threshold: float = Field(16.0, description="MOG2 Mahalanobis distance threshold")
    knn_history: int = Field(500, description="KNN history frames")
    knn_dist2_threshold: float = Field(400.0, description="KNN distance threshold")
    detect_shadows: bool = Field(True, description="Mark shadows in MOG2/KNN baselines")


class DetectionConfig(BaseModel):
    """Configuration for Object Detection."""
    weights: str = Field("yolov8n.pt", description="YOLO weights file or model name")
    conf_threshold: float = Field(0.35, description="Minimum confidence threshold for detections")
    iou_threshold: float = Field(0.45, description="NMS IoU threshold")
    classes: List[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 5, 7],
        description="COCO class IDs for CCTV monitoring (0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck)"
    )
    device: str = Field("cpu", description="Compute device: 'cpu', 'cuda', '0', etc.")


class TrackingConfig(BaseModel):
    """Configuration for Object Tracking."""
    tracking_mode: str = Field("hybrid", description="Tracking mode: 'hybrid' (YOLO + Motion Blob Fallback), 'yolo', 'motion_blob'")
    tracker_type: str = Field("bytetrack.yaml", description="Tracker configuration ('bytetrack.yaml' or 'botsort.yaml')")
    track_buffer: int = Field(30, description="Frames to retain lost tracks before dropping")
    max_history_length: int = Field(60, description="Number of trajectory centroid points to keep for trail rendering")
    min_blob_area: int = Field(150, description="Minimum contour pixel area for motion blob tracking")


class TripwireConfig(BaseModel):
    """Defines a virtual counting line."""
    name: str = Field("Main Gate", description="Identifier for this tripwire")
    pt1: Tuple[float, float] = Field((0.05, 0.5), description="Start coordinate (x, y) - normalized ratio or pixel")
    pt2: Tuple[float, float] = Field((0.95, 0.5), description="End coordinate (x, y) - normalized ratio or pixel")
    color: Tuple[int, int, int] = Field((0, 255, 255), description="Line color in BGR")


class AnalyticsConfig(BaseModel):
    """Configuration for Analytics: Line-crossing, Zones, Heatmaps, and CDRE."""
    enable_tripwires: bool = Field(True, description="Enable virtual tripwire line-crossing counting")
    tripwires: List[TripwireConfig] = Field(default_factory=list)

    enable_heatmap: bool = Field(False, description="Accumulate pedestrian/vehicle motion heatmaps")
    heatmap_decay: float = Field(0.995, description="Decay rate per frame for accumulated heat")
    heatmap_alpha: float = Field(0.4, description="Blending opacity of heatmap overlay")

    extract_cdre: bool = Field(False, description="Extract frame packet sizes and picture types from bitstream")


class CompressionConfig(BaseModel):
    """Configuration for Stage 3: Compression Impact & ROI Optimization."""
    codec: str = Field("libx265", description="Video codec for compression sweeps ('libx265', 'libx264', 'mp4v')")
    reference_qp: int = Field(18, description="Baseline near-lossless QP reference")
    sweep_qp_levels: List[int] = Field(
        default_factory=lambda: [18, 22, 26, 30, 34, 38, 42, 46, 51],
        description="Standard QP sweep ladder"
    )
    knee_retention_target: float = Field(0.90, description="Retention threshold (90%) for operational knee point")
    roi_qp: int = Field(22, description="Target QP for foreground machine-vision regions")
    bg_qp: int = Field(42, description="Target QP for background non-essential regions")


class VisualizerConfig(BaseModel):
    """Configuration for Display Overlays and HUD."""
    view_mode: str = Field(
        "quad",
        description="View layout: 'quad' (4-way split), 'side_by_side' (raw vs clean), 'hud' (full overlay), 'mask_only'"
    )
    show_segmentation_overlay: bool = Field(False, description="Overlay foreground mask on detections in green")
    overlay_alpha: float = Field(0.35, description="Opacity of foreground mask overlay")
    show_trajectories: bool = Field(True, description="Draw trajectory trails behind tracked objects")
    show_boxes: bool = Field(True, description="Draw bounding boxes and class/ID labels")
    show_hud_stats: bool = Field(True, description="Render top-left telemetry HUD (FPS, latency breakdown, counts)")
    hud_bg_color: Tuple[int, int, int] = Field((20, 20, 20), description="Background color of HUD panel in BGR")


class PipelineConfig(BaseModel):
    """Master Configuration schema uniting all 4 stages of the CCTV Analytics Pipeline."""
    version: str = Field("1.0.0", description="Pipeline specification version")
    name: str = Field("cctv-analytics-default", description="Configuration profile name")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    visualizer: VisualizerConfig = Field(default_factory=VisualizerConfig)


def load_config(config_path: Union[str, Path, None] = None) -> PipelineConfig:
    """Load configuration from a YAML or JSON file. If None, returns default config."""
    if config_path is None:
        return PipelineConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return PipelineConfig(**data)


def save_config(config: PipelineConfig, output_path: Union[str, Path]) -> None:
    """Serialize configuration to a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

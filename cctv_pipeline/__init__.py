"""
CCTV Video Analytics Pipeline
Robust CCTV Video Analytics Under Illumination Changes and Compression Artifacts.
"""

__version__ = "1.0.0"
__author__ = "CHRIST University - CSEDS743E04"

from cctv_pipeline.pipeline import CCTVAnalyticsPipeline
from cctv_pipeline.config.settings import PipelineConfig

__all__ = [
    "CCTVAnalyticsPipeline",
    "PipelineConfig",
    "__version__",
]

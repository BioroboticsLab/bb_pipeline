from pipeline.stages import stage
from pipeline.stages.processing import (
    Decoder,
    ImageReader,
    Localizer,
    LocalizerPreprocessor,
    ResultMerger,
)
from pipeline.stages.stage import PipelineStage
from pipeline.stages.visualization import (
    LocalizerVisualizer,
    ResultCrownVisualizer,
    ResultVisualizer,
    SaliencyVisualizer,
)

Stages = (
    ImageReader,
    LocalizerPreprocessor,
    Localizer,
    Decoder,
    ResultMerger,
    LocalizerVisualizer,
    ResultVisualizer,
    ResultCrownVisualizer,
    SaliencyVisualizer,
)

__all__ = ["stage", "PipelineStage"]

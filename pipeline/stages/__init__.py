from pipeline.stages import stage
from pipeline.stages.stage import PipelineStage

from pipeline.stages.processing import ImageReader, LocalizerPreprocessor, \
    Localizer, Decoder, CoordinateMapper, ResultMerger, \
    TagSimilarityEncoder
from pipeline.stages.visualization import LocalizerVisualizer, ResultVisualizer, \
    ResultCrownVisualizer, SaliencyVisualizer


Stages = (ImageReader,
          LocalizerPreprocessor,
          Localizer,
          Decoder,
          CoordinateMapper,
          ResultMerger,
          TagSimilarityEncoder,
          LocalizerVisualizer,
          ResultVisualizer,
          ResultCrownVisualizer,
          SaliencyVisualizer)

__all__ = ["stage", "PipelineStage"]

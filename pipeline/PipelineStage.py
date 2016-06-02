#!/usr/bin/python3

from pipeline.PipelineObject import *


class PipelineStage(object):
    requires = []
    provides = []


class ImageReader(PipelineStage):
    requires = [Filename]
    provides = [Image, Timestamp]


class LocalizerPreprocessor(PipelineStage):
    requires = [Image]
    provides = [LocalizerInputImage]


class Localizer(PipelineStage):
    requires = [LocalizerInputImage]
    provides = [Regions, SaliencyImage, Saliencies, Candidates]


class Decoder(PipelineStage):
    requires = [Regions, Candidates]
    provides = [Positions, Orientations, IDs]


class CoordinateMapper(PipelineStage):
    requires = [Positions]
    provides = [HivePositions]


class ResultMerger(PipelineStage):
    requires = [Positions, HivePositions, Orientations, IDs, Saliencies]
    provides = [PipelineResult]


class TagSimilarityEncoder(PipelineStage):
    requires = [Candidates]
    provides = [Descriptors]


class LocalizerVisualizer(PipelineStage):
    requires = [Image, Regions]
    provides = [CandidateOverlay]


class ResultVisualizer(PipelineStage):
    requires = [CandidateOverlay, Regions, IDs]
    provides = [FinalResultOverlay]


Stages = (ImageReader,
          LocalizerPreprocessor,
          Localizer,
          Decoder,
          CoordinateMapper,
          ResultMerger,
          TagSimilarityEncoder,
          LocalizerVisualizer,
          ResultVisualizer)

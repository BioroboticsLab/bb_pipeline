#!/usr/bin/python3

from os.path import isfile

from cv2 import createCLAHE
import numpy as np
from skimage.io import imread

from bb_binary import parse_image_fname

from pipeline.PipelineObject import *


class PipelineStage(object):
    requires = []
    provides = []


class ImageReader(PipelineStage):
    requires = [Filename]
    provides = [Image, Timestamp, CameraIndex]

    def __call__(self, fname):
        assert(isfile(fname.fname))
        image = imread(fname.fname)
        camIdx, dt = parse_image_fname(fname.fname)
        return [Image(image), Timestamp(dt), CameraIndex(camIdx)]


class LocalizerPreprocessor(PipelineStage):
    requires = [Image]
    provides = [LocalizerInputImage]

    def __init__(self):
        # TODO: add some kind of config module
        clip_limit = 2
        tag_width = 64
        tag_heigth = 64

        self.clahe = createCLAHE(clip_limit, (tag_width, tag_heigth))

    def __call__(self, image):
        return LocalizerInputImage(self.clahe.apply(image.image))


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

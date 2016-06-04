#!/usr/bin/python3

import logging
from os.path import isfile
import sys

from cv2 import createCLAHE
import numpy as np
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter1d

# TODO: don't print keras import messages
stderr = sys.stderr                     # noqa
sys.stderr = open('/dev/null', 'w')     # noqa

import localizer
import localizer.util
import localizer.config
from localizer.visualization import get_roi_overlay
from localizer.localizer import Localizer as LocalizerAPI
from keras.models import model_from_json
sys.stderr = stderr                     # noqa

from bb_binary import parse_image_fname

from pipeline.objects import DecoderRegions, Filename, Image, Timestamp, \
    CameraIndex, Positions, HivePositions, Orientations, IDs, Saliencies, \
    PipelineResult, Candidates, CandidateOverlay, FinalResultOverlay, \
    Regions, Descriptors, LocalizerInputImage, SaliencyImage


class PipelineStage(object):
    requires = []
    provides = []

    def __init__(self, **config):
        pass


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

    def __init__(self, clahe_clip_limit=2, tag_width=64, tag_heigth=64,
                 **config):
        self.clahe = createCLAHE(clahe_clip_limit, (tag_width, tag_heigth))

    def __call__(self, image):
        return [LocalizerInputImage(self.clahe.apply(image.image))]


class Localizer(PipelineStage):
    requires = [LocalizerInputImage]
    provides = [Regions, SaliencyImage, Saliencies, Candidates]

    def __init__(self, saliency_model_path: str,
                 saliency_threshold=0.5,
                 **config):
        self.saliency_threshold = saliency_threshold
        self.localizer = LocalizerAPI()
        self.localizer.logger.setLevel(logging.WARNING)
        self.localizer.load_weights(saliency_model_path)
        self.localizer.compile()

    def __call__(self, input):
        results = self.localizer.detect_tags(
            input.image, saliency_threshold=self.saliency_threshold)
        saliencies, candidates, rois, saliency_image = results

        # TODO: investigate source of offset
        offset = 6
        candidates -= offset

        return [Regions(rois),
                SaliencyImage(saliency_image),
                Saliencies(saliencies),
                Candidates(candidates)]


class DecoderPreprocessor(PipelineStage):
    requires = [Image, Candidates]
    provides = [DecoderRegions]

    def __call__(self, image, candidates):
        image_blurred = gaussian_filter1d(
            gaussian_filter1d(image.image, 2/3, axis=-1), 2 / 3, axis=-2)

        rois, mask = localizer.util.extract_rois(candidates.candidates, image_blurred)

        # TODO: add localizer/decoder roi size difference to config
        return [DecoderRegions((rois[:, :, 18:-18, 18:-18] / 255.) * 2 - 1)]


class Decoder(PipelineStage):
    requires = [DecoderRegions, Candidates]
    provides = [Positions, Orientations, IDs]

    def __init__(self, decoder_model_path, decoder_weigths_path, **config):
        self.model = model_from_json(open(decoder_model_path).read())
        self.model.load_weights(decoder_weigths_path)
        # We can't use model.compile because it requires an optimizer and a loss function.
        # Since we only use the model for inference, we call the private function
        # _make_predict_function(). This is exactly what keras would do otherwise the first
        # time model.predict() is called.
        self.model._make_predict_function()

    def __call__(self, regions, candidates):
        ids = self.model.predict(regions.regions)

        # TODO
        positions = candidates.candidates
        orientations = None

        return [Positions(positions),
                Orientations(orientations),
                IDs(np.array(ids).T[0])]


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
    requires = [Image, Candidates]
    provides = [CandidateOverlay]

    def __call__(self, image, candidates):
        overlay = get_roi_overlay(candidates.candidates, image.image / 255.)

        return [CandidateOverlay(overlay)]


class ResultVisualizer(PipelineStage):
    requires = [CandidateOverlay, Regions, IDs]
    provides = [FinalResultOverlay]


Stages = (ImageReader,
          LocalizerPreprocessor,
          Localizer,
          DecoderPreprocessor,
          Decoder,
          CoordinateMapper,
          ResultMerger,
          TagSimilarityEncoder,
          LocalizerVisualizer,
          ResultVisualizer)
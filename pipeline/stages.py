#!/usr/bin/python3

import logging

import cv2
import numpy as np
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter1d

import localizer
import localizer.util
import localizer.config
from localizer.visualization import get_roi_overlay
from localizer.localizer import Localizer as LocalizerAPI
from keras.models import model_from_json

from bb_binary import parse_image_fname

from pipeline.objects import DecoderRegions, Filename, Image, Timestamp, \
    CameraIndex, Positions, HivePositions, Orientations, IDs, Saliencies, \
    PipelineResult, Candidates, CandidateOverlay, FinalResultOverlay, \
    Regions, Descriptors, LocalizerInputImage, SaliencyImage, \
    PaddedImage, PaddedCandidates


class PipelineStage(object):
    requires = []
    provides = []

    def __init__(self, **config):
        pass

    def __call__(self, *inputs):
        assert len(self.requires) == len(inputs)
        for required, input in zip(self.requires, inputs):
            if hasattr(required, 'validate'):
                required.validate(input)

        outputs = self.call(*inputs)
        if type(outputs) not in [tuple, list]:
            assert len(self.provides) == 1, "If there are multiple outputs, "\
                "then they must be returned as list or tuple! But got {}.".format(outputs)
            outputs = (outputs, )

        assert len(self.provides) == len(outputs)
        for provided, output in zip(self.provides, outputs):
            if hasattr(provided, 'validate'):
                provided.validate(output)
        return outputs

    def call(self, *inputs):
        raise NotImplemented()


class ImageReader(PipelineStage):
    requires = [Filename]
    provides = [Image, Timestamp, CameraIndex]

    def call(self, fname):
        image = imread(fname)
        camIdx, dt = parse_image_fname(fname, 'beesbook')
        return image, dt, camIdx


class LocalizerPreprocessor(PipelineStage):
    requires = [Image]
    provides = [PaddedImage, LocalizerInputImage]

    def __init__(self,
                 clahe_clip_limit=2,
                 clahe_tile_width=64,
                 clahe_tile_heigth=64,
                 **config):
        self.clahe = cv2.createCLAHE(clahe_clip_limit, (clahe_tile_width, clahe_tile_heigth))

    @staticmethod
    def pad(image):
        return np.pad(image, [s // 2 for s in localizer.config.data_imsize], mode='edge')

    def call(self, image):
        return [LocalizerPreprocessor.pad(image),
                LocalizerPreprocessor.pad(self.clahe.apply(image))]


class Localizer(PipelineStage):
    requires = [LocalizerInputImage]
    provides = [Regions, SaliencyImage, Saliencies, Candidates, PaddedCandidates]

    def __init__(self,
                 saliency_model_path,
                 saliency_threshold=0.5,
                 **config):
        self.saliency_threshold = saliency_threshold
        self.localizer = LocalizerAPI()
        self.localizer.logger.setLevel(logging.WARNING)
        self.localizer.load_weights(saliency_model_path)
        self.localizer.compile()

    def call(self, image):
        results = self.localizer.detect_tags(
            image, saliency_threshold=self.saliency_threshold)
        saliencies, candidates, rois, saliency_image = results

        # TODO: investigate source of offset
        offset = 4
        candidates -= offset

        padded_candidates = np.copy(candidates)

        assert(localizer.config.data_imsize[0] == localizer.config.data_imsize[1])
        candidates -= localizer.config.data_imsize[0] // 2

        return [rois, saliency_image, saliencies, candidates, padded_candidates]


class DecoderPreprocessor(PipelineStage):
    requires = [PaddedImage, PaddedCandidates]
    provides = [DecoderRegions]

    def call(self, image, candidates):
        rois, mask = localizer.util.extract_rois(candidates,
                                                 image)
        assert(len(rois) == len(candidates))
        rois = gaussian_filter1d(
            gaussian_filter1d(rois, 2/3, axis=-1), 2 / 3, axis=-2)
        # TODO: add localizer/decoder roi size difference to config
        return (rois[:, :, 18:-18, 18:-18] / 255.) * 2 - 1


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

    def call(self, regions, candidates):
        predicitions = self.model.predict(regions)
        ids = predicitions[:12]
        z_rot = np.arctan2(predicitions[12], predicitions[13])
        y_rot = predicitions[14]
        x_rot = predicitions[15]
        orientations = np.hstack((z_rot, y_rot, x_rot))
        # TODO: use offset from decoder net
        positions = candidates
        return [positions, orientations, np.array(ids).T[0]]


class CoordinateMapper(PipelineStage):
    requires = [Positions]
    provides = [HivePositions]

    def call(self, pos):
        # TODO: map coordinates
        return pos


class ResultMerger(PipelineStage):
    requires = [Positions, HivePositions, Orientations, IDs, Saliencies]
    provides = [PipelineResult]

    def call(self, positions, hive_positions, orientations, ids, saliencies):
        return PipelineResult(
            positions,
            hive_positions,
            orientations,
            ids,
            saliencies
        )


class TagSimilarityEncoder(PipelineStage):
    requires = [Candidates]
    provides = [Descriptors]


class LocalizerVisualizer(PipelineStage):
    requires = [Image, Candidates]
    provides = [CandidateOverlay]

    def call(self, image, candidates):
        overlay = get_roi_overlay(candidates, image / 255.)
        return overlay


class ResultVisualizer(PipelineStage):
    requires = [CandidateOverlay, Candidates, Orientations, IDs]
    provides = [FinalResultOverlay]

    @staticmethod
    def draw_arrow(overlay, z_rotation, position,
                   vis_line_width,
                   vis_arrow_length,
                   vis_arrow_color):
        x_to = np.round(position[0] + vis_arrow_length * np.cos(z_rotation)).astype(np.int32)
        y_to = np.round(position[1] + vis_arrow_length * np.sin(z_rotation)).astype(np.int32)
        cv2.arrowedLine(overlay, tuple(position), (x_to, y_to),
                        vis_arrow_color, vis_line_width, cv2.LINE_AA)

    @staticmethod
    def draw_text(overlay, ids, position,
                  vis_text_color,
                  vis_text_line_width,
                  vis_text_offset,
                  vis_font_size):
        ids = ''.join([str(int(np.round(c))) for c in list(ids)])
        cv2.putText(overlay, ids, tuple(position + np.array(vis_text_offset)),
                    cv2.FONT_HERSHEY_TRIPLEX, vis_font_size,
                    vis_text_color, vis_text_line_width, cv2.LINE_AA)

    def call(self, overlay, candidates, orientations, ids,
             vis_arrow_length=150,
             vis_line_width=6,
             vis_font_size=1.5,
             vis_text_color=(0, 0, 0),
             vis_text_line_width=4,
             vis_text_offset=(-180, -60),
             vis_arrow_color=(255, 0, 0),
             **config):
        overlay = (np.copy(overlay) * 255).astype(np.uint8)
        for idx in range(len(candidates)):
            pos = candidates[idx, ::-1].astype(np.int32)
            ResultVisualizer.draw_text(overlay, ids[idx], pos,
                                       vis_text_color, vis_text_line_width,
                                       vis_text_offset, vis_font_size)
            ResultVisualizer.draw_arrow(overlay, orientations[idx, 0], pos,
                                        vis_line_width, vis_arrow_length, vis_arrow_color)
        return overlay


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

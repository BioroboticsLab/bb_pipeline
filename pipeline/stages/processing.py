import logging

import cv2
import numpy as np
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter1d
import localizer
import localizer.util
import localizer.config
from localizer.localizer import Localizer as LocalizerAPI
from keras.models import model_from_json
from bb_binary import parse_image_fname

from pipeline.stages.stage import PipelineStage
from pipeline.objects import DecoderRegions, Filename, Image, Timestamp, \
    CameraIndex, Positions, HivePositions, Orientations, IDs, Saliencies, \
    PipelineResult, Candidates, Regions, Descriptors, LocalizerInputImage, \
    SaliencyImage, PaddedImage, PaddedCandidates


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
                 clahe_tile_heigth=64):
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

    def __init__(self, weights_path, threshold=0.5):
        self.saliency_threshold = threshold
        self.localizer = LocalizerAPI()
        self.localizer.logger.setLevel(logging.WARNING)
        self.localizer.load_weights(weights_path)
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

    def __init__(self, model_path, weights_path):
        self.model = model_from_json(open(model_path).read())
        self.model.load_weights(weights_path)
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
    requires = [Regions]
    provides = [Descriptors]

    def __init__(self, **config):
        self.model = model_from_json(open(config['model_path']).read())
        self.model.load_weights(config['weights_path'])
        # We can't use model.compile because it requires an optimizer and a loss function.
        # Since we only use the model for inference, we call the private function
        # _make_predict_function(). This is exactly what keras would do otherwise the first
        # time model.predict() is called.
        self.model._make_predict_function()

    @staticmethod
    def bit_array_to_int(a):
        out = 0
        for bit in a:
            out = (out << 1) | int(bit)
        return out

    def call(self, regions):
        # crop images to match input shape of model
        _, _, lx, ly = regions.shape
        _, _, mx, my = self.model.input_shape
        slice_x = slice(lx//2 - mx//2, lx//2 + mx//2)
        slice_y = slice(ly//2 - my//2, ly//2 + my//2)
        regions = regions[:, :, slice_x, slice_y]
        predictions = self.model.predict(regions)
        # thresholding predictions
        predictions = np.sign(predictions)
        predictions = np.where(predictions == 0, -1, predictions)
        predictions = (predictions + 1) * 0.5

        predictions = np.array([TagSimilarityEncoder.bit_array_to_int(pred)
                                for pred in predictions])
        return [predictions]

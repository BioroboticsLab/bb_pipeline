import numbers

import cv2
import composer.core as composer
import numpy as np
from skimage.exposure import equalize_hist
from skimage.feature import peak_local_max
from skimage.io import imread
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from bb_binary import parse_image_fname
from diktya.func_api_helpers import load_model, predict_wrapper
from diktya.distributions import DistributionCollection
from pipeline.stages.stage import PipelineStage
from pipeline.objects import CamParameter, DecoderRegions, Filename, Image, Timestamp, \
    CameraIndex, Positions, HivePositions, Orientations, IDs, Saliencies, \
    PipelineResult, Candidates, Regions, Descriptors, LocalizerInputImage, \
    SaliencyImage, PaddedImage, PaddedCandidates, LocalizerShapes, Radii, \
    DecoderPredictions


class ImageReader(PipelineStage):
    requires = [Filename]
    provides = [Image, Timestamp, CameraIndex]

    def call(self, fname):
        image = imread(fname)
        camIdx, dt = parse_image_fname(fname)
        return image, dt.timestamp(), camIdx


class LocalizerPreprocessor(PipelineStage):
    requires = [Image]
    provides = [PaddedImage, LocalizerInputImage, LocalizerShapes]

    def __init__(self,
                 roi_size=100,
                 downsampled_size=32,
                 clahe_clip_limit=2,
                 clahe_tile_width=64,
                 clahe_tile_heigth=64):
        self.clahe = cv2.createCLAHE(clahe_clip_limit, (clahe_tile_width, clahe_tile_heigth))
        self.roi_size = roi_size
        self.downsampled_size = downsampled_size
        self.pad_size = int(roi_size / downsampled_size * (downsampled_size // 2))

    def pad(self, image):
        return np.pad(image, self.pad_size, mode='edge')

    def call(self, image):
        shapes = {
            'roi_size': self.roi_size,
            'downsampled_size': self.downsampled_size,
            'pad_size': self.pad_size
        }

        return [self.pad(image),
                self.pad(self.clahe.apply(image)),
                shapes]


class Localizer(PipelineStage):
    requires = [LocalizerInputImage, PaddedImage, LocalizerShapes]
    provides = [Regions, SaliencyImage, Saliencies, Candidates, PaddedCandidates]

    def __init__(self,
                 model_path,
                 threshold=0.6):
        self.saliency_threshold = threshold
        self.model = load_model(model_path)
        self.model._make_predict_function()

    @staticmethod
    def extract_saliencies(candidates, saliency):
        saliencies = np.zeros((len(candidates), 1))
        for idx, (r, c) in enumerate(candidates):
            saliencies[idx] = saliency[r, c]
        return saliencies

    @staticmethod
    def extract_rois(candidates, image, roi_size):
        roi_shape = (roi_size, roi_size)
        rois = []
        mask = np.zeros((len(candidates),), dtype=np.bool_)
        for idx, (r, c) in enumerate(candidates):
            rh = roi_shape[0] / 2
            ch = roi_shape[1] / 2
            # probably introducing a bias here
            roi_orig = image[int(np.ceil(r - rh)):int(np.ceil(r + rh)),
                             int(np.ceil(c - ch)):int(np.ceil(c + ch))]
            if roi_orig.shape == roi_shape:
                rois.append(roi_orig)
                mask[idx] = 1
        if not rois:
            raise Exception("No rois found")
        rois = np.stack(rois, axis=0)[:, np.newaxis]
        return rois, mask

    def get_candidates(self, saliency, dist):
        assert (isinstance(dist, numbers.Integral))
        dist = int(dist)
        below_thresh = saliency < self.saliency_threshold
        im = saliency.copy()
        im[below_thresh] = 0.
        candidates = peak_local_max(im, min_distance=dist)
        return candidates

    def call(self, image, orig_image, shapes):
        roi_size = shapes['roi_size']
        downsampled_size = shapes['downsampled_size']
        pad_size = shapes['pad_size']

        downscale_factor = downsampled_size / roi_size
        zoom_shape = [downscale_factor, downscale_factor]
        scale_factor = roi_size / downsampled_size

        image_downsampled = zoom(image, zoom_shape).astype(np.float32) / 255.

        saliency = self.model.predict(image_downsampled[np.newaxis, np.newaxis, :, :])[0, 0]
        saliency = gaussian_filter(saliency, sigma=3/4)

        # 64 is tag size, 2 * 2 due to downsampling in saliency network
        candidates = self.get_candidates(saliency, dist=int(64 / (2 * 2 * scale_factor) - 1))
        saliencies = self.extract_saliencies(candidates, saliency)

        # TODO: investigate source of offset
        # probably a bias in the localizer train data caused by the old pipeline
        offset = 6
        # simulate reverse padding and downsampling of saliency network
        padded_candidates = (((candidates + 2) * 2 + 2) * 2 + 2) * scale_factor + offset
        candidates_img = padded_candidates - pad_size

        rois, mask = self.extract_rois(padded_candidates, orig_image, roi_size)
        rois = rois.astype(np.float32) / 255.

        return [rois, saliency, saliencies, candidates_img, padded_candidates]


class DecoderPreprocessor(PipelineStage):
    requires = [Regions, LocalizerShapes]
    provides = [DecoderRegions]

    def call(self, rois, shapes):
        roi_size = shapes['roi_size']
        # decoder expects input shape [samples, 1, 64, 64]
        crop_size = (roi_size - 64) // 2

        cropped_rois = rois[:, :, crop_size:-crop_size, crop_size:-crop_size]
        cropped_rois = np.stack([equalize_hist(roi) for roi in cropped_rois])

        return cropped_rois * 2 - 1


class Decoder(PipelineStage):
    requires = [DecoderRegions, Candidates]
    provides = [Positions, Orientations, IDs, Radii, DecoderPredictions]

    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.distribution = DistributionCollection.from_hdf5(model_path)
        self._predict = predict_wrapper(self.model.predict, self.model.output_names)
        self.model._make_predict_function()

    def predict(self, regions):
        structed_array = np.zeros((len(regions),),
                                  dtype=self.distribution.norm_dtype)
        predictions = self._predict(regions)
        for name, arr in predictions.items():
            if name.startswith('bit_'):
                bit_idx = int(name.split('_')[1])
                assert arr.shape == (len(regions), 1)
                structed_array['bits'][:, bit_idx] = 2*arr[:, 0] - 1
            else:
                structed_array[name] = arr
        return structed_array

    def call(self, regions, candidates):
        predictions_norm = self.predict(regions)
        predictions = self.distribution.denormalize(predictions_norm)
        ids = predictions['bits']
        z_rot = predictions['z_rotation']
        y_rot = predictions['y_rotation']
        x_rot = predictions['x_rotation']
        orientations = np.hstack((z_rot, y_rot, x_rot))
        positions = candidates + predictions['center']
        radii = predictions['radius']
        return [positions, orientations, ids, radii, predictions]


class CoordinateMapper(PipelineStage):
    requires = [Positions, CamParameter, CameraIndex]
    provides = [HivePositions]

    def call(self, pos, cam_param, camIdx):
        c = composer.Composer()
        c.load_arguments(cam_param)
        pos[:, [0, 1]] = pos[:, [1, 0]]  # swapping y, x -> x,y
        pos = np.array([pos])
        pos = c.map_coordinate(pos, camIdx)
        return pos[0]


class ResultMerger(PipelineStage):
    requires = [Positions, HivePositions, Orientations, IDs, Saliencies, Radii]
    provides = [PipelineResult]

    def call(self, positions, hive_positions, orientations, ids, saliencies, radii):
        return PipelineResult(
            positions,
            hive_positions,
            orientations,
            ids,
            saliencies,
            radii
        )


class TagSimilarityEncoder(PipelineStage):
    requires = [Regions]
    provides = [Descriptors]

    def __init__(self, **config):
        self.model = load_model(config['model_path'])
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

import cv2
import numbers
import numpy as np
from skimage.exposure import equalize_hist
from skimage.feature import peak_local_max
from skimage.io import imread
from scipy.ndimage import zoom as scipy_zoom
from scipy.ndimage.filters import gaussian_filter
from bb_binary import parse_image_fname
from keras.models import load_model
import tensorflow as tf
from pipeline.stages.stage import PipelineStage
from pipeline.objects import Filename, Image, Timestamp, \
    CameraIndex, Positions, Orientations, IDs, BeeSaliencies, \
    PipelineResult, BeeLocalizerPositions, LocalizerInputImage, \
    BeeSaliencyImage, PaddedImage, LocalizerShapes, \
    DecoderPredictions, TagLocalizerPositions, TagSaliencies, \
    TagSaliencyImage, BeeRegions, TagRegions

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


def zoom(image, zoom_factor, gpu=True):
    if not gpu:
        zoom_shape = [zoom_factor, zoom_factor]
        image_zoomed = scipy_zoom(image, zoom_shape, order=3)
        return image_zoomed.astype(np.float32) / 255.

    if not image.dtype == np.float32:
        assert image.dtype == np.uint8
        image = image.astype(np.float32) / 255

    assert image.ndim == 2

    input_shape = (1, image.shape[0], image.shape[1], 1)
    target_shape = (np.round(np.array(image.shape) * (zoom_factor)))
    target_shape = target_shape.astype(np.int)
    img = tf.placeholder(tf.float32, shape=input_shape, name='original_image')
    img_zoom = tf.image.resize_bicubic(img, target_shape)

    processed = session.run(
        img_zoom[0, :, :, 0], feed_dict={img: image[None, :, :, None]})

    return processed


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
                 downsampled_size=100,
                 use_clahe=False,
                 clahe_clip_limit=2,
                 clahe_tile_width=64,
                 clahe_tile_heigth=64):
        if use_clahe:
            self.clahe = cv2.createCLAHE(clahe_clip_limit,
                                         (clahe_tile_width, clahe_tile_heigth))
        else:
            self.clahe = None

        self.roi_size = roi_size
        self.downsampled_size = downsampled_size
        self.pad_size = int(
            roi_size / downsampled_size * (downsampled_size // 2))

    def pad(self, image):
        return np.pad(image, self.pad_size, mode='edge')

    def call(self, image):
        shapes = {
            'roi_size': self.roi_size,
            'downsampled_size': self.downsampled_size,
            'pad_size': self.pad_size
        }

        localizer_input = self.clahe.apply(
            image) if self.clahe is not None else image
        return [self.pad(image), self.pad(localizer_input), shapes]


class Localizer(PipelineStage):
    requires = [LocalizerInputImage, PaddedImage, LocalizerShapes]
    provides = [
        TagRegions,
        TagSaliencyImage,
        TagSaliencies,
        TagLocalizerPositions,
        BeeRegions,
        BeeSaliencyImage,
        BeeSaliencies,
        BeeLocalizerPositions,
    ]

    def __init__(self, model_path, threshold_tag=0.575, threshold_bee=.7):
        self.saliency_threshold_tag = float(threshold_tag)
        self.saliency_threshold_bee = float(threshold_bee)
        self.model = load_model(model_path)
        self.model._make_predict_function()

    @staticmethod
    def extract_saliencies(positions, saliency):
        saliencies = np.zeros((len(positions), 1))
        for idx, (r, c) in enumerate(positions):
            saliencies[idx] = saliency[r, c]
        return saliencies

    @staticmethod
    def extract_rois(positions, image, roi_size):
        roi_shape = (roi_size, roi_size)
        rois = []
        mask = np.zeros((len(positions), ), dtype=np.bool_)
        for idx, (r, c) in enumerate(positions):
            rh = roi_shape[0] / 2
            ch = roi_shape[1] / 2
            # probably introducing a bias here
            roi_orig = image[int(np.ceil(r - rh)):int(np.ceil(r + rh)),
                             int(np.ceil(c - ch)):int(np.ceil(c + ch))]
            if roi_orig.shape == roi_shape:
                rois.append(roi_orig)
                mask[idx] = 1
        if len(rois) > 0:
            rois = np.stack(rois, axis=0)[:, np.newaxis]
        else:
            rois = np.empty(shape=(0, 0, 0, 0))
        return rois, mask

    def get_positions(self, saliency, dist, threshold):
        assert (isinstance(dist, numbers.Integral))
        dist = int(dist)
        below_thresh = saliency < threshold
        im = saliency.copy()
        im[below_thresh] = 0.
        positions = peak_local_max(im, min_distance=dist)
        return positions

    def process_saliency(self, saliency, scale_factor, pad_size,
                         roi_size, orig_image, threshold, sigma=3/4):
        saliency = gaussian_filter(saliency[0, :, :, 0], sigma=sigma)

        # 32 is tag size, 2 * 2 due to downsampling in saliency network
        positions_down = self.get_positions(
            saliency,
            dist=int(32 / (2 * 2 * scale_factor) - 1),
            threshold=threshold)
        saliencies = self.extract_saliencies(positions_down, saliency)

        # TODO: investigate source of offset
        # probably a bias in the localizer train data caused by the
        # old pipeline
        offset = -5
        # simulate reverse padding and downsampling of saliency network
        padded_positions = (((((positions_down + 3) * 2) + 2) * 2 + 3) * 2) * \
            scale_factor + offset

        positions_img = padded_positions - pad_size

        rois, mask = self.extract_rois(padded_positions, orig_image, roi_size)
        rois = rois.astype(np.float32) / 255.

        return [rois, saliency, saliencies, positions_img]

    def call(self, image, orig_image, shapes):
        roi_size = shapes['roi_size']
        downsampled_size = shapes['downsampled_size']
        pad_size = shapes['pad_size']

        downscale_factor = downsampled_size / roi_size
        scale_factor = roi_size / downsampled_size

        if roi_size != downsampled_size:
            image_downsampled = zoom(image, downscale_factor)
        else:
            image_downsampled = image.astype(np.float32) / 255

        saliencies = self.model.predict(image_downsampled[None, :, :, None])

        tag_results = list(self.process_saliency(
            saliencies[0].copy(), scale_factor, pad_size, roi_size,
            orig_image, self.saliency_threshold_tag)
        )
        bee_results = list(self.process_saliency(
            saliencies[1].copy(), scale_factor, pad_size, roi_size,
            orig_image, self.saliency_threshold_bee)
        )

        return tag_results + bee_results


class Decoder(PipelineStage):
    requires = [TagRegions, TagLocalizerPositions]
    provides = [Positions, Orientations, IDs, DecoderPredictions]

    types = np.dtype([('bits', np.float32, (12, )),
                      ('x_rotation', np.float32, (2, )),
                      ('y_rotation', np.float32, (2, )),
                      ('z_rotation', np.float32, (2, )),
                      ('center', np.float32, (2, ))])

    def __init__(self, model_path, use_hist_equalization=True):
        self.model = load_model(model_path)
        self.uses_hist_equalization = use_hist_equalization

        self.model._make_predict_function()

    def preprocess(self, regions):
        cropped_rois = regions[:, :, 34:-34, 34:-34]

        if self.uses_hist_equalization:
            cropped_rois = np.stack(
                [equalize_hist(roi) for roi in cropped_rois])

        return cropped_rois[:, 0, :, :, None]

    def predict(self, regions):
        predictions = self.model.predict(self.preprocess(regions))

        struct = np.empty(len(predictions[0]), dtype=self.types)
        struct['bits'] = np.stack(predictions[:12], -1)[:, 0, :]
        struct['x_rotation'] = predictions[12]
        struct['y_rotation'] = predictions[13]
        struct['z_rotation'] = predictions[14]
        struct['center'] = predictions[15]

        return struct

    def call(self, regions, positions):
        if len(regions) > 0:
            predictions = self.predict(regions)

            bee_ids = predictions['bits']
            z_rot = np.arctan2(predictions['z_rotation'][:, 1],
                               predictions['z_rotation'][:, 0])
            y_rot = np.arctan2(predictions['y_rotation'][:, 1],
                               predictions['y_rotation'][:, 0])
            x_rot = np.arctan2(predictions['x_rotation'][:, 1],
                               predictions['x_rotation'][:, 0])

            return [
                positions + predictions['center'],
                np.stack((z_rot, y_rot, x_rot), axis=1), bee_ids, predictions
            ]
        else:
            return [np.empty(shape=(0, )) for _ in range(len(self.provides))]


class ResultMerger(PipelineStage):
    requires = [Positions, Orientations, IDs, TagSaliencies]
    provides = [PipelineResult]

    def call(self, positions, orientations, ids, saliencies):
        return PipelineResult(positions, orientations, ids, saliencies)

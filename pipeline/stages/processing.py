import json
import numbers

import cv2
import h5py
import numpy as np
import skimage
import tensorflow as tf
from scipy.ndimage import zoom as scipy_zoom
from skimage.exposure import equalize_hist
from skimage.feature import peak_local_max
from skimage.io import imread
from tensorflow.keras.models import load_model

from pipeline.objects import (
    BeeLocalizerPositions,
    BeeRegions,
    BeeSaliencies,
    BeeTypes,
    CameraIndex,
    DecoderPredictions,
    Filename,
    IDs,
    Image,
    LocalizerInputImage,
    LocalizerShapes,
    Orientations,
    PaddedImage,
    PipelineResult,
    Positions,
    SaliencyImages,
    TagLocalizerPositions,
    TagRegions,
    TagSaliencies,
    Timestamp,
)
from pipeline.stages.stage import PipelineStage

from bb_binary import parse_image_fname

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def zoom(image, zoom_factor, gpu=True):
    if not gpu:
        zoom_shape = [zoom_factor, zoom_factor]
        image_zoomed = scipy_zoom(image, zoom_shape, order=3)
        return image_zoomed.astype(np.float32) / 255.0

    if not image.dtype == np.float32:
        assert image.dtype == np.uint8
        image = image.astype(np.float32) / 255

    assert image.ndim == 2

    input_shape = (1, image.shape[0], image.shape[1], 1)
    target_shape = np.round(np.array(image.shape) * (zoom_factor))
    target_shape = target_shape.astype(np.int)
    img = tf.placeholder(tf.float32, shape=input_shape, name="original_image")
    img_zoom = tf.image.resize_bicubic(img, target_shape)

    processed = get_tensorflow_session().run(
        img_zoom[0, :, :, 0], feed_dict={img: image[None, :, :, None]}
    )

    return processed


class InitializedPipelineStage(PipelineStage):
    def __init__(self):
        super().__init__()


class ImageReader(InitializedPipelineStage):
    requires = [Filename]
    provides = [Image, Timestamp, CameraIndex]

    def __init__(self):
        super().__init__()

    def call(self, fname):
        image = imread(fname)
        camIdx, dt = parse_image_fname(fname)
        return image, dt.timestamp(), camIdx


class LocalizerPreprocessor(InitializedPipelineStage):
    requires = [Image]
    provides = [PaddedImage, LocalizerInputImage, LocalizerShapes]

    def __init__(
        self,
        roi_size=100,
        downsampled_size=100,
        use_clahe=False,
        clahe_clip_limit=2,
        clahe_tile_width=64,
        clahe_tile_heigth=64,
    ):
        super().__init__()

        if use_clahe:
            self.clahe = cv2.createCLAHE(
                clahe_clip_limit, (clahe_tile_width, clahe_tile_heigth)
            )
        else:
            self.clahe = None

        self.roi_size = roi_size
        self.downsampled_size = downsampled_size
        self.pad_size = int(roi_size / downsampled_size * (downsampled_size // 2))

    def pad(self, image):
        return np.pad(image, self.pad_size, mode="edge")

    def call(self, image):
        shapes = {
            "roi_size": self.roi_size,
            "downsampled_size": self.downsampled_size,
            "pad_size": self.pad_size,
        }

        localizer_input = self.clahe.apply(image) if self.clahe is not None else image
        return [self.pad(image), self.pad(localizer_input), shapes]


class Localizer(InitializedPipelineStage):
    requires = [LocalizerInputImage, PaddedImage, LocalizerShapes]
    provides = [
        SaliencyImages,
        TagRegions,
        TagSaliencies,
        TagLocalizerPositions,
        BeeRegions,
        BeeSaliencies,
        BeeLocalizerPositions,
        BeeTypes,
    ]

    def __init__(self, model_path, thresholds={}):
        super().__init__()
        self.model = load_model(model_path, compile=False)
        self.model._make_predict_function()

        with h5py.File(model_path, "r") as f:
            self.class_labels = list(f["labels"])
            self.thresholds = dict(
                list(zip(self.class_labels, f["default_thresholds"]))
            )

        if isinstance(thresholds, str):
            thresholds = json.loads(thresholds)

        for k in thresholds.keys():
            self.thresholds[k] = thresholds[k]

    @staticmethod
    def extract_saliencies(positions, saliency):
        saliencies = np.zeros((len(positions), 1))
        for idx, (r, c) in enumerate(positions):
            saliencies[idx] = saliency[int(np.round(r)), int(np.round(c))]
        return saliencies

    @staticmethod
    def extract_rois(positions, image, roi_size):
        roi_shape = (roi_size, roi_size)
        rois = []
        mask = np.zeros((len(positions),), dtype=np.bool_)
        for idx, (r, c) in enumerate(positions):
            rh = roi_shape[0] / 2
            ch = roi_shape[1] / 2
            # probably introducing a bias here
            roi_orig = image[
                int(np.ceil(r - rh)) : int(np.ceil(r + rh)),
                int(np.ceil(c - ch)) : int(np.ceil(c + ch)),
            ]
            if roi_orig.shape == roi_shape:
                rois.append(roi_orig)
                mask[idx] = 1
        if len(rois) > 0:
            rois = np.stack(rois, axis=0)[:, np.newaxis]
        else:
            rois = np.empty(shape=(0, 0, 0, 0))
        return rois, mask

    @staticmethod
    def get_positions(saliency, dist, threshold):
        assert isinstance(dist, numbers.Integral)
        dist = int(dist)
        below_thresh = saliency < threshold
        im = saliency.copy()
        im[below_thresh] = 0.0
        positions = peak_local_max(im, min_distance=dist)
        return positions

    @staticmethod
    def get_subpixel_offsets(saliency, position, subpixel_range):
        sample = saliency[
            position[0] - subpixel_range : position[0] + subpixel_range,
            position[1] - subpixel_range : position[1] + subpixel_range,
        ]

        M = skimage.measure.moments(sample)
        centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

        return (
            centroid[0] - (subpixel_range - 0.5),
            centroid[1] - (subpixel_range - 0.5),
        )

    @staticmethod
    def get_predicted_positions(
        saliency,
        threshold,
        min_distance=1,
        padding=128,
        subpixel_precision=True,
        subpixel_range=3,
    ):
        positions = skimage.feature.peak_local_max(
            saliency, min_distance=min_distance, threshold_abs=threshold
        )

        if subpixel_precision:
            saliency_padded = np.pad(
                saliency, pad_width=subpixel_range, constant_values=0
            )
            subpixel_offsets = [
                Localizer.get_subpixel_offsets(
                    saliency_padded, p + subpixel_range, subpixel_range=subpixel_range
                )
                for p in positions
            ]

            positions = positions.astype(np.float32)
            for idx in range(len(positions)):
                positions[idx, 0] += subpixel_offsets[idx][0]
                positions[idx, 1] += subpixel_offsets[idx][1]

        padded_positions = (((positions + 5) * 2 + 1) * 2 + 1) * 2 + 1

        predictions_positions = padded_positions - padding

        return padded_positions, predictions_positions, positions

    def process_saliency(
        self,
        saliency,
        scale_factor,
        pad_size,
        roi_size,
        orig_image,
        threshold,
        sigma=3 / 4,
    ):
        saliency = saliency[0, :, :]

        (
            padded_positions,
            positions_img,
            saliency_positions,
        ) = Localizer.get_predicted_positions(saliency, threshold, padding=pad_size)

        rois, mask = self.extract_rois(padded_positions, orig_image, roi_size)
        rois = rois.astype(np.float32) / 255.0

        saliencies = self.extract_saliencies(saliency_positions, saliency)

        return [rois, saliency, saliencies, positions_img]

    def call(self, image, orig_image, shapes):
        roi_size = shapes["roi_size"]
        downsampled_size = shapes["downsampled_size"]
        pad_size = shapes["pad_size"]

        downscale_factor = downsampled_size / roi_size
        scale_factor = roi_size / downsampled_size

        if roi_size != downsampled_size:
            image_downsampled = zoom(image, downscale_factor)
        else:
            image_downsampled = image.astype(np.float32) / 255

        saliencies = self.model.predict(image_downsampled[None, :, :, None])

        bee_regions = []
        bee_saliencies = []
        bee_positions = []
        bee_types = []
        tag_results = None

        for class_idx, class_label in enumerate(self.class_labels):
            results = list(
                self.process_saliency(
                    saliencies[:, :, :, class_idx].copy(),
                    scale_factor,
                    pad_size,
                    roi_size,
                    orig_image,
                    self.thresholds[class_label],
                )
            )

            if class_label == "MarkedBee":
                tr, _, ts, tp = results
                tag_results = [tr, ts, tp]
            else:
                br, _, bs, bp = results

                if len(br):
                    bee_regions.append(br)
                    bee_saliencies.append(bs)
                    bee_positions.append(bp)
                    bee_types.append([class_label for _ in range(len(br))])

        if len(bee_regions):
            bee_regions = np.concatenate(bee_regions)
            bee_saliencies = np.concatenate(bee_saliencies)
            bee_positions = np.concatenate(bee_positions)
            bee_types = np.concatenate(bee_types)
        else:
            bee_regions = np.ndarray(shape=(0, 0, 0, 0))
            bee_saliencies = np.ndarray(shape=(0, 0, 0, 0))
            bee_positions = np.ndarray(shape=(0, 0, 0))
            bee_types = np.ndarray(shape=(0))

        assert tag_results is not None

        return (
            [saliencies[0]]
            + tag_results
            + [bee_regions, bee_saliencies, bee_positions, bee_types]
        )


class Decoder(InitializedPipelineStage):
    requires = [TagRegions, TagLocalizerPositions]
    provides = [Positions, Orientations, IDs, DecoderPredictions]

    types = np.dtype(
        [
            ("bits", np.float32, (12,)),
            ("x_rotation", np.float32, (2,)),
            ("y_rotation", np.float32, (2,)),
            ("z_rotation", np.float32, (2,)),
            ("center", np.float32, (2,)),
        ]
    )

    def __init__(self, model_path, use_hist_equalization=True):
        super().__init__()
        self.model = load_model(model_path, compile=False)
        self.uses_hist_equalization = use_hist_equalization

        self.model._make_predict_function()

    def preprocess(self, regions):
        cropped_rois = regions[:, :, 34:-34, 34:-34]

        if self.uses_hist_equalization:
            cropped_rois = np.stack([equalize_hist(roi) for roi in cropped_rois])
        return cropped_rois[:, 0, :, :, None]

    def predict(self, regions):
        predictions = self.model.predict(self.preprocess(regions))

        struct = np.empty(len(predictions[0]), dtype=self.types)
        struct["bits"] = np.stack(predictions[:12], -1)[:, 0, :]
        struct["x_rotation"] = predictions[12]
        struct["y_rotation"] = predictions[13]
        struct["z_rotation"] = predictions[14]
        struct["center"] = predictions[15]

        return struct

    def call(self, regions, positions):
        if len(regions) > 0:
            predictions = self.predict(regions)

            bee_ids = predictions["bits"]
            z_rot = np.arctan2(
                predictions["z_rotation"][:, 1], predictions["z_rotation"][:, 0]
            )
            y_rot = np.arctan2(
                predictions["y_rotation"][:, 1], predictions["y_rotation"][:, 0]
            )
            x_rot = np.arctan2(
                predictions["x_rotation"][:, 1], predictions["x_rotation"][:, 0]
            )

            return [
                positions + predictions["center"],
                np.stack((z_rot, y_rot, x_rot), axis=1),
                bee_ids,
                predictions,
            ]
        else:
            return [np.empty(shape=(0,)) for _ in range(len(self.provides))]


class ResultMerger(InitializedPipelineStage):
    requires = [
        BeeLocalizerPositions,
        Positions,
        Orientations,
        IDs,
        TagSaliencies,
        BeeSaliencies,
        BeeTypes,
    ]
    provides = [PipelineResult]

    def __init__(self):
        super().__init__()

    def call(
        self,
        bee_positions,
        tag_positions,
        orientations,
        ids,
        tag_saliencies,
        bee_saliencies,
        bee_types,
    ):
        return PipelineResult(
            bee_positions,
            tag_positions,
            orientations,
            ids,
            tag_saliencies,
            bee_saliencies,
            bee_types,
        )

import json

import numpy as np
import skimage
import skimage.exposure
import skimage.transform

import pipeline.pipeline
from pipeline.objects import IDs, Image, Positions, TagSaliencies

import bb_utils


class TagDecoder:
    def __init__(self, localizer_threshold=None, decoder_threshold=0.99):
        config = pipeline.pipeline.get_auto_config()

        self.decoder_threshold = decoder_threshold

        if localizer_threshold is not None:
            config["Localizer"]["thresholds"] = json.dumps(
                dict(MarkedBee=localizer_threshold)
            )

        self.pipeline = pipeline.pipeline.Pipeline(
            [Image], [Positions, TagSaliencies, IDs], **config
        )

    def decode_image(
        self, rescale_factor, image, clahe_size=50, use_clahe=True, order=1
    ):
        image_rescaled = skimage.transform.rescale(
            image, 1 / rescale_factor, order=order
        )

        if use_clahe:
            image_rescaled = skimage.exposure.equalize_adapthist(
                np.clip(image_rescaled, 0.0, 1.0), clahe_size
            )

        results = self.pipeline([(image_rescaled * 255).astype(np.uint8)])

        return results

    def brute_force_scaling(self, rescale_factors, image):
        confidences = []

        for rescale_factor in rescale_factors:
            results = self.decode_image(rescale_factor, image)

            if len(results[IDs]) > 0:
                confidences.append(
                    np.mean(np.prod(np.abs(0.5 - results[IDs]) * 2, axis=1))
                )
            else:
                confidences.append(0)

            best_factor_idx = np.argmax(confidences)
            best_factors = rescale_factors[
                max(0, best_factor_idx - 1) : min(
                    len(confidences) - 1, best_factor_idx + 2
                )
            ]

        return (
            np.linspace(best_factors[0], best_factors[-1], num=10),
            confidences[best_factor_idx],
            rescale_factors[best_factor_idx],
        )

    def get_ids(self, results):
        confidences = np.prod(np.abs(0.5 - results[IDs]) * 2, axis=-1)
        high_conf_indices = np.argwhere(confidences > self.decoder_threshold)

        def _get_ferwar_id(idx):
            bbid = bb_utils.ids.BeesbookID.from_bb_binary(results[IDs][idx])
            return bbid.as_ferwar()

        if high_conf_indices.shape[-1] > 0:
            return set(map(_get_ferwar_id, high_conf_indices[:, 0],))
        else:
            return set()

    def __call__(
        self, image, num_brute_force_iterations=3, rescale_factors=None, augment=True
    ):
        if rescale_factors is None:
            rescale_factors = 2 ** np.linspace(-1, 5, num=10)

        for _ in range(num_brute_force_iterations):
            rescale_factors, _, rescale_factor = self.brute_force_scaling(
                rescale_factors, image
            )

        def _get_detected_ids(image, use_clahe=False):
            results = self.decode_image(
                rescale_factor, image, use_clahe=use_clahe, order=3,
            )
            return self.get_ids(results)

        ferwar_ids = set()
        if augment:
            for k in range(4):
                for use_clahe in (False, True):
                    ferwar_ids.update(_get_detected_ids(np.rot90(image, k), use_clahe))
        else:
            ferwar_ids.update(_get_detected_ids(image))

        return ferwar_ids

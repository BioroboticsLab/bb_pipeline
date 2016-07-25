#!/usr/bin/python3

import logging

import cairocffi as cairo
import cv2
import numpy as np
from skimage.io import imread
from skimage.color import hsv2rgb
from skimage.transform import resize
from skimage.exposure import adjust_gamma
from scipy.ndimage.filters import gaussian_filter1d
import localizer
import localizer.util
import localizer.config
from localizer.visualization import get_roi_overlay, get_circle_overlay
from localizer.localizer import Localizer as LocalizerAPI
from keras.models import model_from_json

from bb_binary import parse_image_fname

from pipeline.objects import DecoderRegions, Filename, Image, Timestamp, \
    CameraIndex, Positions, HivePositions, Orientations, IDs, Saliencies, \
    PipelineResult, Candidates, CandidateOverlay, FinalResultOverlay, \
    Regions, Descriptors, LocalizerInputImage, SaliencyImage, \
    PaddedImage, PaddedCandidates, CrownOverlay, SaliencyOverlay


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


class SaliencyVisualizer(PipelineStage):
    requires = [Image, SaliencyImage]
    provides = [SaliencyOverlay]

    def __init__(self, saliency_visualizer_hue=240 / 360.,
                 saliency_visualizer_gamma=0.25,
                 **config):
        self.hue = saliency_visualizer_hue
        self.gamma = saliency_visualizer_gamma

    def call(self, image, saliency_image):
        img_resize = resize(image, saliency_image.shape)
        saliency_range = max(0.15, saliency_image.max() - saliency_image.min())
        saliency_norm = (saliency_image - saliency_image.min()) / saliency_range
        hsv = np.stack([
            self.hue * np.ones_like(saliency_norm),
            adjust_gamma(saliency_norm, gamma=self.gamma),
            img_resize
        ], axis=-1)
        return hsv2rgb(hsv)


class LocalizerVisualizer(PipelineStage):
    requires = [Image, Candidates]
    provides = [CandidateOverlay]

    def __init__(self, roi_overlay='circle', **config):
        assert roi_overlay in ('rect', 'circle')
        self.roi_overlay = roi_overlay

    def call(self, image, candidates):
        if self.roi_overlay == 'rect':
            return get_roi_overlay(candidates, image / 255.)
        else:
            return get_circle_overlay(candidates, image / 255.)


class ResultCrownVisualizer(PipelineStage):
    requires = [Image, Candidates, Orientations, IDs]
    provides = [CrownOverlay]

    def __init__(self, outer_radius=60, inner_radius=32,
                 orientation_radius=72,
                 true_hue=120 / 360., false_hue=240 / 360., orientation_hue=0 / 360.,
                 line_width=3,
                 **config):
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.orientation_radius = orientation_radius
        self.true_hue = true_hue
        self.false_hue = false_hue
        self.orientation_hue = orientation_hue
        self.line_width = line_width

    def call(self, image, candidates, orientations, ids):
        z_rots = orientations[:, 0]
        candidates = np.stack([candidates[:, 1], candidates[:, 0]], axis=-1)
        height, width = image.shape
        # hsva
        # avsh
        image_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(image_surface)
        ctx.set_antialias(cairo.ANTIALIAS_NONE)
        for z, pos, id in zip(z_rots, candidates, ids):
            self._draw_crown(ctx, z, pos, id)
        image_surface.flush()
        overlay = np.ndarray(shape=(height, width, 4),
                             buffer=image_surface.get_data(),
                             dtype=np.uint8)
        overlay_hsva = overlay / 255.
        image_surface.finish()
        return overlay_hsva

    @staticmethod
    def _hsv(h, s, v, alpha=0):
        return (v, s, h, alpha)

    def _draw_crown(self, ctx: cairo.Context, angle, pos, bits):
        def x(w):
            return float(np.cos(w))

        def y(w):
            return float(np.sin(w))

        def arc_path(start, end, inner, outer):
            ctx.new_path()
            ctx.arc(0, 0, inner, start, end)
            ctx.line_to(outer*x(end), outer*y(end))
            ctx.arc_negative(0, 0, outer, end, start)
            ctx.line_to(inner*x(start), inner*y(start))
            ctx.close_path()

        def fill_arc(start, end, color, inner, outer):
            clear_color = (0, 0, 0, 0)
            arc_path(start, end, inner, outer)
            ctx.set_source_rgba(*clear_color)
            ctx.fill_preserve()
            ctx.set_source_rgba(*color)
            ctx.fill()

        def draw_arc_line(start, end, color, inner, outer):
            arc_path(start, end, inner, outer)
            ctx.set_source_rgba(*color)
            ctx.set_line_width(self.line_width)
            ctx.stroke()

        def confidence_to_alpha(c):
            return 2*(c - 0.5)

        ctx.save()
        ctx.translate(pos[0], pos[1])
        ctx.rotate(angle)
        w = 1 / len(bits) * 2 * np.pi
        for i, bit in enumerate(bits):
            if bit > 0.5:
                color = self._hsv(self.true_hue, confidence_to_alpha(bit), 1, 1)
            else:
                color = self._hsv(self.false_hue, confidence_to_alpha(1 - bit), 1, 1)
            fill_arc(w*i, w*(i+1), color, self.inner_radius, self.outer_radius)
        fill_arc(-np.pi / 2, np.pi / 2,
                 self._hsv(self.orientation_hue, 1, 1, 1),
                 self.outer_radius, self.orientation_radius)

        stroke_color = self._hsv(0, 0, 0.5, 1)
        for i in range(len(bits)):
            draw_arc_line(w*i, w*(i+1), stroke_color, self.inner_radius, self.outer_radius)
        ctx.restore()

    @staticmethod
    def add_overlay(image, overlay_hsva):
        height, width = image.shape
        # hsva
        alpha = overlay_hsva[:, :, 3]
        k = 0.75
        increase_mask = alpha >= 0.4
        image[increase_mask] = k*image[increase_mask] + (1 - k)
        image_hsv = np.stack([
            overlay_hsva[:, :, 0],
            overlay_hsva[:, :, 1],
            image,
        ], axis=-1)
        # TODO: this hsv2rgb conversion is super inefficent! As there are crowns
        # only on a friction of the total image, one could set the rgb image
        # to the gray image everywhere. The crowns could then be added with hsv2rgb
        # for every subwindow.
        return hsv2rgb(image_hsv)


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
          ResultVisualizer,
          ResultCrownVisualizer,
          SaliencyVisualizer)

import cv2
import numpy as np
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage.transform import resize
from skimage.exposure import adjust_gamma
import matplotlib

from pipeline.stages.stage import PipelineStage
from pipeline.objects import Image, Orientations, IDs, SaliencyOverlay, \
    Candidates, CandidateOverlay, FinalResultOverlay, SaliencyImage, \
    CrownOverlay, LocalizerShapes


class SaliencyVisualizer(PipelineStage):
    requires = [Image, SaliencyImage]
    provides = [SaliencyOverlay]

    def __init__(self, hue=240 / 360., gamma=0.2):
        self.hue = hue
        self.gamma = gamma

    def call(self, image, saliency_image):
        img_resize = resize(image, saliency_image.shape)
        saliency_range = max(0.15, saliency_image.max() - saliency_image.min())
        saliency_norm = (saliency_image - saliency_image.min()) / saliency_range
        saliency_gamma = adjust_gamma(saliency_norm, gamma=self.gamma)
        cmap = matplotlib.cm.get_cmap('viridis')
        cmap_hsv = rgb2hsv(cmap(saliency_gamma)[:, :, :3])
        hsv = np.stack([
            cmap_hsv[:, :, 0],
            saliency_gamma,
            img_resize
        ], axis=-1)
        return hsv2rgb(hsv)


class LocalizerVisualizer(PipelineStage):
    requires = [Image, Candidates, LocalizerShapes]
    provides = [CandidateOverlay]

    def __init__(self, roi_overlay='circle'):
        assert roi_overlay in ('rect', 'circle')
        self.roi_overlay = roi_overlay

    @staticmethod
    def get_roi_overlay(coordinates, image, roi_size):
        def roi_slice(coord):
            return slice(int(np.ceil(coord - roi_size / 2)),
                         int(np.ceil(coord + roi_size / 2)))

        pltim = np.zeros((image.shape[0], image.shape[1], 3))
        pltim[:, :, 0] = image
        pltim[:, :, 1] = image
        pltim[:, :, 2] = image
        for idx, (r, c) in enumerate(coordinates):
            sl_r = roi_slice(r)
            sl_c = roi_slice(c)
            pltim[sl_r, sl_c] = 1.

        return pltim

    @staticmethod
    def get_circle_overlay(coordinates, image, radius=32, line_width=8,
                           color=(1., 0, 0)):
        import cairocffi as cairo

        height, width = image.shape[:2]
        if image.ndim == 2:
            overlay = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 3:
            overlay = image.copy()
        else:
            raise Exception("Did not understand image shape {}.".format(image.shape))

        image_surface = cairo.ImageSurface(cairo.FORMAT_A8, image.shape[1], image.shape[0])
        ctx = cairo.Context(image_surface)
        for x, y in coordinates:
            ctx.save()
            ctx.translate(int(y), int(x))
            ctx.new_path()
            ctx.arc(0, 0, radius + line_width / 2., 0, 2 * np.pi)
            ctx.close_path()
            ctx.set_source_rgba(0, 0, 0, 1)
            ctx.set_line_width(line_width)
            ctx.stroke()
            ctx.restore()

        image_surface.flush()
        circles = np.ndarray(shape=(height, width),
                             buffer=image_surface.get_data(),
                             dtype=np.uint8)
        circles_mask = (circles == 255)
        for i in range(3):
            overlay[circles_mask, i] = color[i]
        image_surface.finish()

        return overlay

    def call(self, image, candidates, shapes):
        roi_size = shapes['roi_size']

        if self.roi_overlay == 'rect':
            return self.get_roi_overlay(candidates, image / 255., roi_size)
        else:
            return self.get_circle_overlay(candidates, image / 255.)


class ResultCrownVisualizer(PipelineStage):
    requires = [Image, Candidates, Orientations, IDs]
    provides = [CrownOverlay]

    def __init__(self, outer_radius=60, inner_radius=32,
                 orientation_radius=72,
                 true_hue=120 / 360., false_hue=240 / 360., orientation_hue=0 / 360.,
                 line_width=3):
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.orientation_radius = orientation_radius
        self.true_hue = true_hue
        self.false_hue = false_hue
        self.orientation_hue = orientation_hue
        self.line_width = line_width

    def call(self, image, candidates, orientations, ids):
        import cairocffi as cairo
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
    def _hsv2rgba(h, s, v, alpha=0):
        r, g, b = hsv2rgb(np.array([[[h, s, v]]])).flatten().tolist()
        return (b, g, r, alpha)

    def _draw_crown(self, ctx, angle, pos, bits):
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
                color = self._hsv2rgba(self.true_hue, confidence_to_alpha(bit), 1, 1)
            else:
                color = self._hsv2rgba(self.false_hue, confidence_to_alpha(1 - bit), 1, 1)
            fill_arc(w*i, w*(i+1), color, self.inner_radius, self.outer_radius)
        fill_arc(-np.pi / 2, np.pi / 2,
                 self._hsv2rgba(self.orientation_hue, 1, 1, 1),
                 self.outer_radius, self.orientation_radius)

        stroke_color = self._hsv2rgba(0, 0, 0.5, 1)
        for i in range(len(bits)):
            draw_arc_line(w*i, w*(i+1), stroke_color, self.inner_radius, self.outer_radius)
        ctx.restore()

    @staticmethod
    def add_overlay(image, overlay):
        alpha = overlay[:, :, 3, np.newaxis]
        image_rgb = gray2rgb(image) * (1 - alpha) + alpha * overlay[:, :, :3]
        return image_rgb


class ResultVisualizer(PipelineStage):
    requires = [CandidateOverlay, Candidates, Orientations, IDs]
    provides = [FinalResultOverlay]

    @staticmethod
    def draw_arrow(overlay, z_rotation, position,
                   line_width,
                   arrow_length,
                   arrow_color):
        x_to = np.round(position[0] + arrow_length * np.cos(z_rotation)).astype(np.int32)
        y_to = np.round(position[1] + arrow_length * np.sin(z_rotation)).astype(np.int32)
        cv2.arrowedLine(overlay, tuple(position), (x_to, y_to),
                        arrow_color, line_width, cv2.LINE_AA)

    def call(self, overlay, candidates, orientations, ids,
             arrow_length=150,
             line_width=6,
             font_size=1.5,
             text_color=(0, 0, 0),
             text_line_width=4,
             text_offset=(-180, -60),
             arrow_color=(255, 0, 0)):
        overlay = (np.copy(overlay) * 255).astype(np.uint8)
        for idx in range(len(candidates)):
            pos = candidates[idx, ::-1].astype(np.int32)
            ResultVisualizer.draw_arrow(overlay, orientations[idx, 0], pos,
                                        line_width, arrow_length, arrow_color)
        return overlay

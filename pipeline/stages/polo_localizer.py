"""
POLO-based bee localizer stage for the BeesBook pipeline.

Uses a TorchScript-exported POLO model (a YOLOv8-based point detector from
mooch443/POLO) for multi-class bee detection. POLO natively outputs
per-detection class predictions, so no separate crop classifier is needed.

Provides the same output contract as the heatmap-based Localizer stage:
  SaliencyImages, TagRegions, TagSaliencies, TagLocalizerPositions,
  BeeRegions, BeeSaliencies, BeeLocalizerPositions, BeeTypes

Requires only PyTorch at runtime (no ultralytics dependency).

Export the POLO .pt model to TorchScript *twice* — once per device — because
torch.jit.trace bakes the device of in-graph tensor constructors into the
saved graph, and a CPU-traced graph errors with "tensors on different device"
when loaded on CUDA:

    from ultralytics import YOLO
    import torch
    for dev in ("cpu", "cuda"):
        model = YOLO("polo26_feedercams.pt")
        m = model.model.to(dev).eval()
        scripted = torch.jit.trace(m, torch.randn(1, 3, 640, 640, device=dev),
                                   check_trace=False)
        torch.jit.save(scripted, f"polo26_feedercams_{dev}.torchscript")

PoloLocalizer picks the right variant at load time based on the resolved
device. Callers pass a base name (with or without _cpu/_cuda suffix and with
or without .torchscript extension) — see _resolve_torchscript_path.
"""

import json
import pathlib

import cv2
import numpy as np
import torch

from pipeline.objects import (
    BeeLocalizerPositions,
    BeeRegions,
    BeeSaliencies,
    BeeTypes,
    LocalizerInputImage,
    LocalizerShapes,
    PaddedImage,
    SaliencyImages,
    TagLocalizerPositions,
    TagRegions,
    TagSaliencies,
)
from pipeline.stages.processing import InitializedPipelineStage, Localizer, point_nms

# Default POLO class names matching standard POLO training.
_DEFAULT_POLO_CLASS_NAMES = {
    0: "UnmarkedBee",
    1: "MarkedBee",
    2: "BeeInCell",
    3: "UpsideDownBee",
}


def _letterbox(img, new_shape=(640, 640)):
    """Resize + pad image to *new_shape*, preserving aspect ratio.

    Matches the ultralytics ``LetterBox`` transform (``auto=False``).

    Returns:
        img_padded: Letterboxed image (*new_shape* HWC).
        scale: Scale factor applied.
        pad: ``(dw, dh)`` float padding offsets for coordinate inversion.
    """
    h, w = img.shape[:2]
    target_h, target_w = new_shape
    scale = min(target_h / h, target_w / w)

    new_unpad_w = int(round(w * scale))
    new_unpad_h = int(round(h * scale))

    dw = (target_w - new_unpad_w) / 2.0
    dh = (target_h - new_unpad_h) / 2.0

    if (new_unpad_h, new_unpad_w) != (h, w):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h),
                         interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return img, scale, (dw, dh)


def _postprocess(pred_tensor, conf_threshold, scale, pad):
    """Decode raw POLO output into detections in original image coordinates.

    Args:
        pred_tensor: ``(1, 2+C, N)`` tensor — ``[x, y, cls0, cls1, ...]``
            in letterbox pixel space.
        conf_threshold: Minimum detection confidence.
        scale: Letterbox scale factor.
        pad: ``(dw, dh)`` letterbox padding.

    Returns:
        positions_xy: ``(M, 2)`` float32 array of ``(x, y)`` in original
            (pre-letterbox) image coordinates.
        confidences: ``(M,)`` float32 confidence scores.
        class_ids: ``(M,)`` int64 class IDs.
    """
    pred = pred_tensor[0].T          # (N, 2+C)
    xy = pred[:, :2]                 # (N, 2)  x, y in letterbox space
    cls_scores = pred[:, 2:]         # (N, C)

    conf, cls_id = cls_scores.max(dim=1)

    mask = conf >= conf_threshold
    xy = xy[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]

    if len(xy) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    dw, dh = pad
    xy_np = xy.cpu().numpy().astype(np.float64)
    xy_np[:, 0] = (xy_np[:, 0] - dw) / scale
    xy_np[:, 1] = (xy_np[:, 1] - dh) / scale

    return (
        xy_np.astype(np.float32),
        conf.cpu().numpy(),
        cls_id.cpu().numpy(),
    )


class PoloLocalizer(InitializedPipelineStage):
    """Multi-class point-detection localizer using a TorchScript POLO model.

    Drop-in replacement for the heatmap-based Localizer. POLO detects bees
    and classifies them (MarkedBee, UnmarkedBee, BeeInCell, UpsideDownBee) in
    a single forward pass. MarkedBee detections are passed to the Decoder for
    barcode reading, just like the heatmap localizer.

    The model is loaded via ``torch.jit.load`` — no ultralytics dependency.
    """

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

    def __init__(self, polo_model_path, attributes_path,
                 confidence_threshold=0.5, imgsz=640, nms_radius=0,
                 polo_class_names=None, device="auto"):
        super().__init__()

        self.device = self._resolve_device(device)
        resolved_path = self._resolve_torchscript_path(polo_model_path, self.device)
        print(f"[POLO] device={self.device.type}, loaded {resolved_path}")
        self.model = torch.jit.load(str(resolved_path), map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._cpu_fallback_triggered = False

        self.confidence_threshold = float(confidence_threshold)
        self.imgsz = int(imgsz)
        self.nms_radius = float(nms_radius)

        # Class-name mapping {int_id: "ClassName"}.
        if polo_class_names is not None:
            if isinstance(polo_class_names, str):
                polo_class_names = json.loads(polo_class_names)
            self.polo_class_names = {
                int(k): v for k, v in polo_class_names.items()
            }
        else:
            self.polo_class_names = dict(_DEFAULT_POLO_CLASS_NAMES)

        with open(attributes_path, 'r') as f:
            attributes = json.load(f)
            self.class_labels = attributes['class_labels']

    @staticmethod
    def _resolve_torchscript_path(polo_model_path, device):
        """Resolve a POLO base path to the device-specific .torchscript file.

        Accepts any of these input forms and always returns the variant that
        matches ``device``:
          - ``.../polo26_feedercams``                    (bare base, preferred)
          - ``.../polo26_feedercams.torchscript``        (legacy, pre-split)
          - ``.../polo26_feedercams_cpu.torchscript``    (explicit — gets re-resolved)
          - ``.../polo26_feedercams_cuda.torchscript``   (explicit — gets re-resolved)

        Falls back to the original path (with a warning) if the device-specific
        file is missing but the original does exist; raises FileNotFoundError
        if neither exists.
        """
        tag = "cuda" if device.type == "cuda" else "cpu"
        p = pathlib.Path(polo_model_path)
        stem = p.stem
        if stem.endswith("_cpu"):
            stem = stem[:-4]
        elif stem.endswith("_cuda"):
            stem = stem[:-5]
        candidate = p.parent / f"{stem}_{tag}.torchscript"
        if candidate.exists():
            return candidate
        if p.exists():
            print(f"[POLO] device-specific variant not found at {candidate}; "
                  f"falling back to {p}")
            return p
        raise FileNotFoundError(
            f"No POLO TorchScript at {candidate} or {p}"
        )

    @staticmethod
    def _resolve_device(device):
        if isinstance(device, torch.device):
            return device
        requested = str(device).strip().lower() if device is not None else "auto"
        if requested in ("", "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[POLO] cuda requested but unavailable; using cpu instead")
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _is_cross_device_torchscript_error(exc):
        msg = str(exc).lower()
        return (
            "expected all tensors to be on the same device" in msg
            and "cuda" in msg
            and "cpu" in msg
        )

    def call(self, image, orig_image, shapes):
        roi_size = shapes["roi_size"]
        pad_size = shapes["pad_size"]

        # POLO expects 3-channel input; replicate grayscale to RGB.
        if image.ndim == 2:
            image_rgb = np.stack([image, image, image], axis=-1)
        else:
            image_rgb = image

        # --- Letterbox preprocessing ---
        target = (self.imgsz, self.imgsz)
        img_lb, scale, pad = _letterbox(image_rgb, target)

        # HWC uint8 → CHW float32 [0, 1] → batched tensor
        img_t = img_lb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_t).unsqueeze(0).to(self.device)

        # --- Forward pass ---
        with torch.no_grad():
            try:
                raw = self.model(img_t)
            except RuntimeError as exc:
                if self.device.type != "cuda" or not self._is_cross_device_torchscript_error(exc):
                    raise

                if not self._cpu_fallback_triggered:
                    print("[POLO] TorchScript CUDA/CPU mismatch; retrying on CPU")
                    self._cpu_fallback_triggered = True
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                img_t = img_t.to(self.device)
                raw = self.model(img_t)
        # TorchScript output is (pred_tensor, feature_maps); keep only [0].
        pred_tensor = raw[0] if isinstance(raw, (tuple, list)) else raw

        # --- Decode predictions ---
        positions_xy, conf, cls_ids = _postprocess(
            pred_tensor, self.confidence_threshold, scale, pad,
        )

        class_names = np.array(
            [self.polo_class_names.get(int(c), "Unknown") for c in cls_ids]
        )

        # Parse detections into (positions, confidences, class_names).
        # positions_xy[:,0]=x (col), positions_xy[:,1]=y (row) in the
        # padded-input-image coordinate space (letterbox was undone above).
        if len(positions_xy) > 0:
            # Convert (x, y) → pipeline (row, col) convention.
            padded_positions = np.stack(
                [positions_xy[:, 1], positions_xy[:, 0]], axis=1,
            )  # (N, 2) as (row, col) in padded-image coords
            positions_img = padded_positions - pad_size  # original image coords
            confidences = conf[:, np.newaxis]            # (N, 1)

            # Cross-class NMS: suppress duplicates at the same location.
            if self.nms_radius > 0:
                keep = point_nms(
                    padded_positions, conf, self.nms_radius,
                    class_names=class_names, class_agnostic=True,
                )
                padded_positions = padded_positions[keep]
                positions_img = positions_img[keep]
                confidences = confidences[keep]
                class_names = class_names[keep]
        else:
            padded_positions = np.empty((0, 2), dtype=np.float32)
            positions_img = np.empty((0, 2), dtype=np.float32)
            confidences = np.zeros((0, 1), dtype=np.float32)
            class_names = np.array([], dtype='<U20')

        # Split detections into MarkedBee (→ tag results for Decoder)
        # and all other classes (→ bee results).
        tag_mask = class_names == "MarkedBee"
        bee_mask = ~tag_mask

        # --- Tag (MarkedBee) results ---
        tag_padded = padded_positions[tag_mask]
        tag_pos = positions_img[tag_mask]
        tag_conf = confidences[tag_mask]

        if len(tag_padded) > 0:
            tag_rois, roi_mask = Localizer.extract_rois(
                tag_padded, orig_image, roi_size
            )
            tag_rois = tag_rois.astype(np.float32) / 255.0
            tag_pos = tag_pos[roi_mask]
            tag_conf = tag_conf[roi_mask]
        else:
            tag_rois = np.empty(shape=(0, 0, 0, 0))

        tag_results = [tag_rois, tag_conf, tag_pos]

        # --- Bee (non-MarkedBee) results ---
        bee_padded = padded_positions[bee_mask]
        bee_pos_all = positions_img[bee_mask]
        bee_conf_all = confidences[bee_mask]
        bee_types_all = class_names[bee_mask]

        if len(bee_padded) > 0:
            bee_rois, roi_mask = Localizer.extract_rois(
                bee_padded, orig_image, roi_size
            )
            bee_rois = bee_rois.astype(np.float32) / 255.0
            bee_positions = bee_pos_all[roi_mask]
            bee_saliencies = bee_conf_all[roi_mask]
            bee_types = bee_types_all[roi_mask]
        else:
            bee_rois = np.ndarray(shape=(0, 0, 0, 0))
            bee_saliencies = np.ndarray(shape=(0, 0, 0, 0))
            bee_positions = np.ndarray(shape=(0, 0, 0))
            bee_types = np.ndarray(shape=(0,))

        # Dummy saliency image — POLO does not produce heatmaps.
        saliency_images = np.zeros(
            (1, 1, len(self.class_labels)), dtype=np.float32
        )

        return (
            [saliency_images]
            + tag_results
            + [bee_rois, bee_saliencies, bee_positions, bee_types]
        )

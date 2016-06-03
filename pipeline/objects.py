#!/usr/bin/python3


class PipelineObject(object):
    pass


class Filename(PipelineObject):
    def __init__(self, fname):
        self.fname = fname


class CameraIndex(PipelineObject):
    def __init__(self, idx):
        self.idx = idx


class Image(PipelineObject):
    def __init__(self, image):
        self.image = image


class Timestamp(PipelineObject):
    def __init__(self, timestamp):
        self.timestamp = timestamp


''' Downsampled and preprocessed '''
class LocalizerInputImage(PipelineObject):
    def __init__(self, image):
        self.image = image


''' Image patches of candidates (original image size) '''
class Regions(PipelineObject):
    def __init__(self, regions):
        self.regions = regions


''' Saliency image (downsampled image coordinates) '''
class SaliencyImage(PipelineObject):
    def __init__(self, image):
        self.image = image


class Saliencies(PipelineObject):
    def __init__(self, saliencies):
        self.saliencies = saliencies


''' Center positions of localized tags (original image coordinates) '''
class Candidates(PipelineObject):
    def __init__(self, candidates):
        self.candidates = candidates


''' Blurred image patches for Decoder '''
class DecoderRegions(PipelineObject):
    def __init__(self, regions):
        self.regions = regions


''' Output of Autoencoder for each Candidate '''
class Descriptors(PipelineObject):
    def __init__(self, descriptors):
        self.descriptors = descriptors


''' Final tag center coordinates (corrected by Decoder) '''
class Positions(PipelineObject):
    def __init__(self, positions):
        self.positions = positions


''' Final tag center in hive coordinate system '''
class HivePositions(PipelineObject):
    def __init__(self, positions):
        self.positions = positions


class Orientations(PipelineObject):
    def __init__(self, orientations):
        self.orientations = orientations


class IDs(PipelineObject):
    def __init__(self, ids):
        self.ids = ids


class PipelineResult(PipelineObject):
    def __init__(self, result):
        self.result = result


class CandidateOverlay(PipelineObject):
    def __init__(self, overlay):
        self.overlay = overlay


class FinalResultOverlay(PipelineObject):
    def __init__(self, overlay):
        self.overlay = overlay

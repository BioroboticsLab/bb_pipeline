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


class LocalizerInputImage(PipelineObject):
    ''' Downsampled and preprocessed '''
    def __init__(self, image):
        self.image = image


class Regions(PipelineObject):
    ''' Image patches of candidates (original image size) '''
    def __init__(self, regions):
        self.regions = regions


class SaliencyImage(PipelineObject):
    ''' Saliency image (downsampled image coordinates) '''
    def __init__(self, image):
        self.image = image


class Saliencies(PipelineObject):
    def __init__(self, saliencies):
        self.saliencies = saliencies


class Candidates(PipelineObject):
    ''' Center positions of localized tags (original image coordinates) '''
    def __init__(self, candidates):
        self.candidates = candidates


class DecoderRegions(PipelineObject):
    ''' Blurred image patches for Decoder '''
    def __init__(self, regions):
        self.regions = regions


class Descriptors(PipelineObject):
    ''' Output of Autoencoder for each Candidate '''
    def __init__(self, descriptors):
        self.descriptors = descriptors


class Positions(PipelineObject):
    ''' Final tag center coordinates (corrected by Decoder) '''
    def __init__(self, positions):
        self.positions = positions


class HivePositions(PipelineObject):
    ''' Final tag center in hive coordinate system '''
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

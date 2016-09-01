import numpy as np
import os


class PipelineObject(object):
    pass


class PipelineResult(PipelineObject):
    def __init__(self, positions, hive_positions, orientations, ids, saliencies, radii):
        self.positions = positions
        self.hive_positions = hive_positions
        self.orientations = orientations
        self.ids = ids
        self.saliencies = saliencies
        self.radii = radii


class PipelineObjectDescription(object):
    type = None

    @classmethod
    def validate(cls, value):
        if type(value) is not cls.type:
            raise Exception(
                "Expected value of be of type {}.  But got value {} of type {}"
                .format(cls.type, value, type(value)))


class FilenameDescription(PipelineObjectDescription):
    type = str

    @classmethod
    def validate(cls, fname):
        super(FilenameDescription, cls).validate(fname)
        if not os.path.isfile(fname):
            raise Exception("Got invalid filename {}.".format(fname))


class NumpyArrayDescription(PipelineObjectDescription):
    type = np.ndarray
    shape = None
    ndim = None

    @classmethod
    def validate(cls, arr):
        def ndim():
            if cls.shape is not None:
                return len(cls.shape)
            else:
                return cls.ndim

        super(NumpyArrayDescription, cls).validate(arr)
        if ndim():
            if ndim() != len(arr.shape):
                raise Exception(
                    "ndim missmatch: Expected {}, got shape {} with {}"
                    .format(ndim(), arr.shape, len(arr.shape)))
        if cls.shape is not None:
            for i, (expected, got) in enumerate(zip(cls.shape, arr.shape)):
                if expected is not None:
                    if expected != got:
                        raise Exception(
                            "Shape missmatch at dimension {}: expected {}, got {}."
                            .format(i, expected, got))


class CameraIndex(PipelineObjectDescription):
    type = int


class Filename(FilenameDescription):
    pass


class Image(NumpyArrayDescription):
    ndim = 2


class PaddedImage(NumpyArrayDescription):
    ndim = 2


class Timestamp(PipelineObjectDescription):
    type = float


class LocalizerShapes(PipelineObjectDescription):
    type = dict


class LocalizerInputImage(NumpyArrayDescription):
    ndim = 2


class Regions(NumpyArrayDescription):
    ''' Image patches of candidates (original image size) '''
    ndim = 4


class SaliencyImage(NumpyArrayDescription):
    ''' Saliency image (downsampled image coordinates) '''
    ndim = 2


class Saliencies(NumpyArrayDescription):
    pass


class Candidates(NumpyArrayDescription):
    ''' Center positions of localized tags (original image coordinates) '''
    pass


class Radii(NumpyArrayDescription):
    ''' Radii of localized tags '''
    pass


class PaddedCandidates(NumpyArrayDescription):
    ''' Center positions of localized tags (padded image coordinates) '''
    pass


class DecoderRegions(NumpyArrayDescription):
    ''' Blurred image patches for Decoder '''
    pass


class Descriptors(NumpyArrayDescription):
    ''' Output of Autoencoder for each Candidate '''
    pass


class Positions(NumpyArrayDescription):
    ''' Final tag center coordinates (corrected by Decoder) '''
    pass


class HivePositions(NumpyArrayDescription):
    ''' Final tag center in hive coordinate system '''
    pass


class Orientations(NumpyArrayDescription):
    pass


class IDs(NumpyArrayDescription):
    pass


class CandidateOverlay(NumpyArrayDescription):
    pass


class CrownOverlay(NumpyArrayDescription):
    pass


class SaliencyOverlay(NumpyArrayDescription):
    pass


class FinalResultOverlay(NumpyArrayDescription):
    pass


class DecoderPredictions(NumpyArrayDescription):
    pass

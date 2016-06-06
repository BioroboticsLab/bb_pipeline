import numpy as np
import os


class PipelineObject(object):
    pass


class PipelineResult(PipelineObject):
    def __init__(self, positions, hive_positions, orientations, ids, saliencies):
        self.positions = positions
        self.hive_positions = hive_positions
        self.orientations = orientations
        self.ids = ids
        self.saliencies = saliencies


class PipelineObjectDescription(object):
    type = None

    @classmethod
    def validate(cls, value):
        assert type(value) is cls.type


class FilenameDescription(PipelineObjectDescription):
    type = str

    @classmethod
    def validate(cls, fname):
        super(FilenameDescription, cls).validate(fname)
        assert os.path.isfile(fname)


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
            assert ndim() == len(arr.shape), \
                "ndim missmatch: Expected {}, got shape {} with {}"\
                .format(ndim(), arr.shape, len(arr.shape))
        if cls.shape is not None:
            for i, (expected, got) in enumerate(zip(cls.shape, arr.shape)):
                if expected is not None:
                    assert expected == got, \
                        "Shape missmatch at dimension {}: expected {}, got {}."\
                        .format(i, expected, got)


class CameraIndex(PipelineObjectDescription):
    type = int


class Filename(FilenameDescription):
    pass


class Image(NumpyArrayDescription):
    ndim = 2


class Timestamp(PipelineObjectDescription):
    type = float


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


class FinalResultOverlay(NumpyArrayDescription):
    pass

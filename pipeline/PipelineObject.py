#!/usr/bin/python3


class PipelineObject(object):
    pass

class Filename(PipelineObject):
    pass

class Image(PipelineObject):
    pass

class Timestamp(PipelineObject):
    pass

''' Downsampled and preprocessed '''
class LocalizerInputImage(PipelineObject):
    pass

''' List of ROIs in original image coordinates '''
class Regions(PipelineObject):
    pass

''' Upsampled saliency (original image coordinates) '''
class SaliencyImage(PipelineObject):
    pass

class Saliencies(PipelineObject):
    pass

''' Image patches of candidates (original image size) '''
class Candidates(PipelineObject):
    pass

''' Output of Autoencoder for each Candidate '''
class Descriptors(PipelineObject):
    pass

''' Final tag center coordinates (corrected by Decoder) '''
class Positions(PipelineObject):
    pass

''' Final tag center in hive coordinate system '''
class HivePositions(PipelineObject):
    pass

class Orientations(PipelineObject):
    pass

class IDs(PipelineObject):
    pass

class PipelineResult(PipelineObject):
    pass


class CandidateOverlay(PipelineObject):
    pass

class FinalResultOverlay(PipelineObject):
    pass

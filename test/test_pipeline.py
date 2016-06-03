#!/usr/bin/python3

import os

from pipeline import Pipeline
from pipeline.object import *
from pipeline.stage import *


def test_empty_pipeline():
    pipeline = Pipeline([], [])
    assert(len(pipeline.pipeline) == 0)


def test_stages_are_instantiated():
    pipeline = Pipeline([Image], [LocalizerInputImage])
    assert(all([issubclass(type(stage), PipelineStage) for stage in pipeline.pipeline]))


def _assert_types(actual, expected):
    for actual_type, expected_type in zip(actual, expected):
        assert(type(actual_type) == expected_type)


def test_simple_pipeline():
    pipeline = Pipeline([Filename], [SaliencyImage, Descriptors])

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer,
                       TagSimilarityEncoder]
    _assert_types(pipeline.pipeline, expected_stages)


def test_imagereader():
    pipeline = Pipeline([Filename], [Image, Timestamp, CameraIndex])

    expected_stages = [ImageReader]
    _assert_types(pipeline.pipeline, expected_stages)

    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'

    outputs = pipeline([Filename(fname)])
    assert(len(outputs) == 3)

    expected_outputs = [Image, Timestamp, CameraIndex]
    _assert_types(outputs, expected_outputs)

    assert(outputs[0].image.shape == (3000, 4000))
    assert(outputs[1].timestamp.year == 2015)
    assert(outputs[1].timestamp.month == 8)
    assert(outputs[1].timestamp.day == 21)
    assert(outputs[1].timestamp.hour == 16)
    assert(outputs[1].timestamp.minute == 15)
    assert(outputs[1].timestamp.second == 30)
    assert(outputs[1].timestamp.microsecond == 884267)
    assert(outputs[2].idx == 2)


def test_localizer():
    pipeline = Pipeline([Filename], [Regions, Candidates])

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer]
    _assert_types(pipeline.pipeline, expected_stages)

    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'

    outputs = pipeline([Filename(fname)])

    expected_outputs = [Regions, Candidates]
    _assert_types(outputs, expected_outputs)

    regions = outputs[0]
    assert(len(regions.regions) > 0)

    candidates = outputs[1]
    assert(len(regions.regions) == len(candidates.candidates))

    for candidate in candidates.candidates:
        assert(candidate[0] >= 0 and candidate[0] < 3000)
        assert(candidate[1] >= 0 and candidate[1] < 4000)


def test_decoder():
    pipeline = Pipeline([Filename], [Candidates, IDs])

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer,
                       DecoderPreprocessor,
                       Decoder]
    _assert_types(pipeline.pipeline, expected_stages)

    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'

    outputs = pipeline([Filename(fname)])

    expected_outputs = [Candidates, IDs]
    _assert_types(outputs, expected_outputs)

    candidates = outputs[0]
    ids = outputs[1]

    assert(len(ids.ids) == len(candidates.candidates))

    print()
    for pos, id in zip(candidates.candidates, ids.ids):
        pos = np.round(pos).astype(np.int)
        id = ''.join([str(int(b)) for b in (np.round(id))])
        print('Detection at ({}, {}) \t ID: {}'.format(pos[0], pos[1], id))



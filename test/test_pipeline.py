#!/usr/bin/python3

import os

from pipeline import Pipeline
from pipeline.PipelineObject import *
from pipeline.PipelineStage import *


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
    assert(len(outputs))

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






#!/usr/bin/python3

from pipeline import Pipeline
from pipeline.PipelineObject import *
from pipeline.PipelineStage import *


def test_empty_pipeline():
    pipeline = Pipeline([], [])
    assert(len(pipeline.pipeline) == 0)


def test_stages_are_instantiated():
    pipeline = Pipeline([Image], [LocalizerInputImage])
    assert(all([issubclass(type(stage), PipelineStage) for stage in pipeline.pipeline]))


def test_simple_pipeline():
    pipeline = Pipeline([Filename], [SaliencyImage, Descriptors])

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer,
                       TagSimilarityEncoder]

    for stage, expected_stage in zip(pipeline.pipeline, expected_stages):
        assert(type(stage) == expected_stage)

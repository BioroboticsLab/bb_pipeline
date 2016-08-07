from pipeline.objects import PipelineObjectDescription, CameraIndex, \
    NumpyArrayDescription
import pytest
import numpy as np


def test_simple_objects():
    assert issubclass(CameraIndex, PipelineObjectDescription)
    assert CameraIndex.type == int
    CameraIndex.validate(10)
    CameraIndex.validate(-10)
    CameraIndex.validate(0)
    with pytest.raises(Exception):
        CameraIndex.validate(0.013)


def test_numpy_descriptions():
    class WithShape(NumpyArrayDescription):
        shape = (None, 10)

    WithShape.validate(np.zeros((100, 10)))

    with pytest.raises(Exception):
        WithShape.validate(np.zeros((10, 5)))

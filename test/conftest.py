
import pytest
import os
import pickle
from pipeline import Pipeline
from pipeline.pipeline import get_auto_config
from pipeline.objects import Filename, LocalizerPositions, IDs, Saliencies, \
    Orientations, Image, SaliencyImage


def get_test_fname(name):
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, name)


@pytest.fixture
def bees_image():
    return get_test_fname('data/Cam_2_20150821161530_884267.jpeg')


@pytest.fixture
def bee_in_the_center_image():
    return get_test_fname('data/a_bee_in_the_center.jpeg')


@pytest.fixture
def bees_video():
    return get_test_fname(
        'data/Cam_0_20150821161642_833382_TO_Cam_0_20150821161648_253846.mkv')


@pytest.fixture
def filelists_path():
    return get_test_fname('data/filelists')


@pytest.fixture
def bees_video_2016():
    return get_test_fname(
        'data/Cam_0_2016-07-19T18:21:33.097618Z--2016-07-19T18:21:34.092604Z.mkv')


@pytest.fixture
def pipeline_config():
    return get_auto_config()


@pytest.fixture
def pipeline_results(pipeline_config, bees_image, outdir):
    bees_image_name, _ = os.path.splitext(os.path.basename(bees_image))
    output_fname = outdir.join(bees_image_name + "_pipeline_output.pickle")
    if output_fname.exists():
        with open(str(output_fname), "rb") as f:
            outputs = pickle.load(f)
    else:
        pipeline = Pipeline([Filename],
                            [LocalizerPositions, IDs, Saliencies, Orientations,
                             Image, SaliencyImage],
                            **pipeline_config)
        outputs = pipeline([bees_image])
        with open(str(output_fname), "wb") as f:
            pickle.dump(outputs, f)
    return outputs


@pytest.fixture
def outdir():
    from py.path import local
    path = local("test").join("out")
    if not path.ensure(dir=True):
        path.mkdir()
    return path

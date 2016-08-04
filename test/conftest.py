
import pytest
import os
import pickle
from pipeline import Pipeline

from pipeline.objects import Filename, Candidates, IDs, Saliencies, Orientations, Image, \
    SaliencyImage


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
    return get_test_fname('data/Cam_0_20150821161642_833382_TO_Cam_0_20150821161648_253846.mkv')


@pytest.fixture
def filelists_path():
    return get_test_fname('data/filelists')


@pytest.fixture
def pipeline_config():
    saliency_weights = get_test_fname('models/localizer/saliency-weights.hdf5')
    decoder_weights = get_test_fname('models/decoder/decoder_weights.hdf5')
    decoder_model = get_test_fname('models/decoder/decoder_architecture.json')
    tagSimilarityEncoder_model = get_test_fname('models/tagSimilarityEncoder/tagSimilarityEncoder_architecture.json')
    tagSimilarityEncoder_weights = get_test_fname('models/tagSimilarityEncoder/tagSimilarityEncoder_weights.h5')

    for fname in (saliency_weights, decoder_model, decoder_weights, tagSimilarityEncoder_model, tagSimilarityEncoder_weights):
        assert os.path.exists(fname), \
            "Not found {}. Did you forgot to run `./get_test_models.sh`?".format(fname)

    return {
        'Localizer': {
            'model_path': saliency_weights,
        },
        'Decoder': {
            'model_path': decoder_model,
            'weights_path': decoder_weights,
        },
        'TagSimilarityEncoder': {
            'model_path': tagSimilarityEncoder_model,
            'weights_path': tagSimilarityEncoder_weights
        }
    }


@pytest.fixture
def pipeline_results(pipeline_config, bees_image, outdir):
    bees_image_name, _ = os.path.splitext(os.path.basename(bees_image))
    output_fname = outdir.join(bees_image_name + "_pipeline_output.pickle")
    if output_fname.exists():
        with open(str(output_fname), "rb") as f:
            outputs = pickle.load(f)
    else:
        pipeline = Pipeline([Filename],
                            [Candidates, IDs, Saliencies, Orientations, Image,
                             SaliencyImage],
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

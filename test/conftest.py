
import pytest
import os


def get_test_fname(name):
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, name)


@pytest.fixture
def bees_image():
    return get_test_fname('data/Cam_2_20150821161530_884267.jpeg')


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
    for fname in (saliency_weights, decoder_model, decoder_weights):
        assert os.path.exists(fname), \
            "Not found {}. Did you forgot to run `./get_test_models.sh`?".format(fname)

    return {
        'saliency_model_path': saliency_weights,
        'decoder_model_path': decoder_model,
        'decoder_weigths_path': decoder_weights,
    }

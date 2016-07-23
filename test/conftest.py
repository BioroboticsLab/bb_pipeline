
import pytest
import os
import json
import numpy as np
import scipy.misc

from pipeline import Pipeline

from pipeline.objects import Filename, Candidates, IDs, Saliencies, Orientations, Image


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
    for fname in (saliency_weights, decoder_model, decoder_weights):
        assert os.path.exists(fname), \
            "Not found {}. Did you forgot to run `./get_test_models.sh`?".format(fname)

    return {
        'saliency_model_path': saliency_weights,
        'decoder_model_path': decoder_model,
        'decoder_weigths_path': decoder_weights,
    }


@pytest.fixture
def pipeline_results(pipeline_config, bees_image, outdir):
    objs = [Filename, Candidates, IDs, Saliencies, Orientations, Image]
    objs_dict = {o.__name__: o for o in objs}

    def to_builtin(outputs):
        def np_to_list(x):
            if type(x) == np.ndarray:
                return x.tolist()
            else:
                return x
        return {cls.__name__: np_to_list(o) for cls, o in outputs.items()}

    def to_numpy(outputs):
        return {objs_dict[name]: np.array(list_arr) for name, list_arr in outputs.items()}

    bees_image_name, _ = os.path.splitext(os.path.basename(bees_image))
    output_fname = outdir.join(bees_image_name + "_pipeline_output.json")
    if output_fname.exists():
        with open(str(output_fname)) as f:
            outputs = to_numpy(json.load(f))
    else:
        pipeline = Pipeline([Filename],
                            [Candidates, IDs, Saliencies, Orientations],
                            **pipeline_config)
        outputs = pipeline([bees_image])
        with open(str(output_fname), "w+") as f:
            json.dump(to_builtin(outputs), f)

    outputs[Image] = scipy.misc.imread(bees_image)
    return outputs


@pytest.fixture
def outdir():
    from py.path import local
    path = local("test").join("out")
    if not path.ensure(dir=True):
        path.mkdir()
    return path

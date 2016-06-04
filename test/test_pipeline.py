import os
import pytest
import time
import datetime
import pytz

from scipy.misc import imread
import numpy as np

from pipeline import Pipeline
from pipeline.stages import Localizer, PipelineStage, ImageReader, \
    LocalizerPreprocessor, TagSimilarityEncoder, Decoder, DecoderPreprocessor

from pipeline.objects import DecoderRegions, Filename, Image, Timestamp, \
    CameraIndex, Positions, HivePositions, Orientations, IDs, Saliencies, \
    PipelineResult, Candidates, CandidateOverlay, FinalResultOverlay, \
    Regions, Descriptors, LocalizerInputImage, SaliencyImage

from bb_binary import Repository


def get_test_fname(name):
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, name)


@pytest.fixture
def bees_image():
    return get_test_fname('data/Cam_2_20150821161530_884267.jpeg')


@pytest.fixture
def config():
    saliency_weights = get_test_fname('models/localizer/')
    decoder_weights = get_test_fname('models/decoder/decoder_weights.hdf5')
    decoder_model = get_test_fname('models/decoder/decoder_architecture.hdf5')
    for fname in (saliency_weights, decoder_model, decoder_weights):
        assert os.path.exists(fname), \
            "Not found {}. Did you forgot to run `./get_test_models.sh`?".format(fname)

    return {
        'saliency_model_path': saliency_weights,
        'decoder_model_path': decoder_model,
        'decoder_weigths_path': decoder_weights,
    }


def test_empty_pipeline():
    pipeline = Pipeline([], [])
    assert(len(pipeline.pipeline) == 0)


def test_stages_are_instantiated(config):
    pipeline = Pipeline([Image], [LocalizerInputImage], **config)
    assert(all([issubclass(type(stage), PipelineStage) for stage in pipeline.pipeline]))


def _assert_types(actual, expected):
    for actual_type, expected_type in zip(actual, expected):
        assert(type(actual_type) == expected_type)


def test_simple_pipeline(config):
    pipeline = Pipeline([Filename], [SaliencyImage, Descriptors], **config)

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer,
                       TagSimilarityEncoder]
    _assert_types(pipeline.pipeline, expected_stages)


def test_imagereader(bees_image, config):
    pipeline = Pipeline([Filename], [Image, Timestamp, CameraIndex], **config)

    expected_stages = [ImageReader]
    _assert_types(pipeline.pipeline, expected_stages)

    outputs = pipeline([Filename(bees_image)])
    assert(len(outputs) == 3)

    expected_outputs = [Image, Timestamp, CameraIndex]
    _assert_types([outputs[Image], outputs[Timestamp], outputs[CameraIndex]], expected_outputs)

    im = outputs[Image]
    ts = outputs[Timestamp]
    idx = outputs[CameraIndex]

    print(ts.timestamp)
    tz = pytz.timezone('Europe/Berlin')
    dt = datetime.datetime.fromtimestamp(ts.timestamp, tz=pytz.utc)
    dt = dt.astimezone(tz)
    assert(im.image.shape == (3000, 4000))
    assert(dt.year == 2015)
    assert(dt.month == 8)
    assert(dt.day == 21)
    assert(dt.hour == 16)
    assert(dt.minute == 15)
    assert(dt.second == 30)
    assert(dt.microsecond == 884267)
    assert(idx.idx == 2)


def test_localizer(config):
    pipeline = Pipeline([Filename], [Regions, Candidates], **config)

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer]
    _assert_types(pipeline.pipeline, expected_stages)

    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'

    outputs = pipeline([Filename(fname)])

    expected_outputs = [Regions, Candidates]
    _assert_types([outputs[Regions], outputs[Candidates]], expected_outputs)

    regions = outputs[Regions]
    assert(len(regions.regions) > 0)

    candidates = outputs[Candidates]
    assert(len(regions.regions) == len(candidates.candidates))

    for candidate in candidates.candidates:
        assert(candidate[0] >= 0 and candidate[0] < 3000)
        assert(candidate[1] >= 0 and candidate[1] < 4000)


def test_decoder(config):
    pipeline = Pipeline([Filename], [Candidates, IDs], **config)

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer,
                       DecoderPreprocessor,
                       Decoder]
    _assert_types(pipeline.pipeline, expected_stages)

    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'

    outputs = pipeline([Filename(fname)])

    expected_outputs = [Candidates, IDs]
    _assert_types([outputs[Candidates], outputs[IDs]], expected_outputs)

    candidates = outputs[Candidates]
    ids = outputs[IDs]

    assert(len(ids.ids) == len(candidates.candidates))

    print()
    for pos, id in zip(candidates.candidates, ids.ids):
        pos = np.round(pos).astype(np.int)
        id = ''.join([str(int(b)) for b in (np.round(id))])
        print('Detection at ({}, {}) \t ID: {}'.format(pos[0], pos[1], id))


def test_print_config_dict(config):
    pipeline = Pipeline([Filename], [PipelineResult], **config)
    config_dict_str = pipeline._config_dict()
    config_dict = eval(config_dict_str)
    print(config_dict)
    assert 'decoder_model_path' in config_dict
    assert config_dict['decoder_model_path'] == 'REQUIRED'
    assert config_dict['decoder_weigths_path'] == 'REQUIRED'
    assert config_dict['saliency_model_path'] == 'REQUIRED'
    assert 'tag_heigth' in config_dict
    assert 'tag_width' in config_dict
    assert 'clahe_clip_limit' in config_dict
    assert 'saliency_threshold' in config_dict

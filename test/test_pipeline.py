import os
import time
import datetime
import pytz
import pytest

from scipy.misc import imread, imsave
from scipy.ndimage.interpolation import zoom

import numpy as np

from pipeline import Pipeline
from pipeline.pipeline import GeneratorProcessor, get_auto_config
from pipeline.io import BBBinaryRepoSink, video_generator
from pipeline.stages import Localizer, PipelineStage, ImageReader, \
    LocalizerPreprocessor, TagSimilarityEncoder, Decoder, \
    ResultCrownVisualizer, LocalizerVisualizer, SaliencyVisualizer

from pipeline.objects import Filename, Image, Timestamp, CameraIndex, IDs, \
    PipelineResult, LocalizerPositions, Regions, Descriptors, LocalizerInputImage, \
    SaliencyImage, PaddedLocalizerPositions, PaddedImage, Orientations, LocalizerShapes, \
    Radii
from bb_binary import Repository, DataSource, FrameContainer


def test_empty_pipeline():
    pipeline = Pipeline([], [])
    assert(len(pipeline.pipeline) == 0)


def test_stages_are_instantiated(pipeline_config):
    pipeline = Pipeline([Image], [LocalizerInputImage], **pipeline_config)
    assert(all([issubclass(type(stage), PipelineStage) for stage in pipeline.pipeline]))


def _assert_types(actual, expected):
    for actual_type, expected_type in zip(actual, expected):
        assert(type(actual_type) == expected_type)


def test_simple_pipeline(pipeline_config):
    pipeline = Pipeline([Filename], [SaliencyImage, Descriptors], **pipeline_config)

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer,
                       TagSimilarityEncoder]
    _assert_types(pipeline.pipeline, expected_stages)


def test_imagereader(bees_image, pipeline_config):
    pipeline = Pipeline([Filename], [Image, Timestamp, CameraIndex], **pipeline_config)

    expected_stages = [ImageReader]
    _assert_types(pipeline.pipeline, expected_stages)

    outputs = pipeline([bees_image])
    assert(len(outputs) == 3)

    assert Image in outputs
    assert Timestamp in outputs
    assert CameraIndex in outputs

    im = outputs[Image]
    ts = outputs[Timestamp]
    idx = outputs[CameraIndex]

    tz = pytz.timezone('Europe/Berlin')
    dt = datetime.datetime.fromtimestamp(ts, tz=pytz.utc)
    dt = dt.astimezone(tz)
    assert(im.shape == (3000, 4000))
    assert(dt.year == 2015)
    assert(dt.month == 8)
    assert(dt.day == 21)
    assert(dt.hour == 16)
    assert(dt.minute == 15)
    assert(dt.second == 30)
    assert(dt.microsecond == 884267)
    assert(idx == 2)


def test_localizer(pipeline_config):
    pipeline = Pipeline([Filename], [Regions, LocalizerPositions], **pipeline_config)

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer]
    _assert_types(pipeline.pipeline, expected_stages)

    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'

    outputs = pipeline([fname])

    assert len(outputs) == 2
    assert Regions in outputs
    assert LocalizerPositions in outputs

    regions = outputs[Regions]
    assert(len(regions) > 0)

    positions = outputs[LocalizerPositions]
    assert(len(regions) == len(positions))

    for pos in positions:
        assert(pos[0] >= 0 and pos[0] < 3000)
        assert(pos[1] >= 0 and pos[1] < 4000)


def test_padding(pipeline_config):
    pipeline = Pipeline([Filename], [PaddedLocalizerPositions, LocalizerPositions,
                                     PaddedImage, Image, LocalizerShapes],
                        **pipeline_config)

    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'
    outputs = pipeline([fname])

    assert len(outputs) == 5
    assert PaddedImage in outputs
    assert PaddedLocalizerPositions in outputs

    offset = outputs[LocalizerShapes]['roi_size'] // 2
    for padded, original in zip(outputs[PaddedLocalizerPositions],
                                outputs[LocalizerPositions]):
        assert(all([(pc - offset) == oc for pc, oc in zip(padded, original)]))


@pytest.mark.slow
def test_decoder(pipeline_config):
    pipeline = Pipeline([Filename], [LocalizerPositions, IDs, Radii], **pipeline_config)

    expected_stages = [ImageReader,
                       LocalizerPreprocessor,
                       Localizer,
                       Decoder]
    _assert_types(pipeline.pipeline, expected_stages)

    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'

    outputs = pipeline([fname])

    assert len(outputs) == 3
    assert IDs in outputs
    assert LocalizerPositions in outputs
    assert Radii in outputs

    positions = outputs[LocalizerPositions]
    ids = outputs[IDs]
    radii = outputs[Radii]

    assert(len(ids) == len(positions))
    assert(len(ids) == len(radii))

    for pos, id, radius in zip(positions, ids, radii):
        pos = np.round(pos).astype(np.int)
        id = ''.join([str(int(b)) for b in (np.round(id))])
        print('Detection at ({}, {}) \t ID: {} \t Radius: {}'.format(pos[0], pos[1], id, radius))


def test_tagSimilarityEncoder(pipeline_config):
    pipeline = Pipeline([Filename], [Descriptors], **pipeline_config)
    fname = os.path.dirname(__file__) + '/data/Cam_2_20150821161530_884267.jpeg'

    outputs = pipeline([fname])
    assert Descriptors in outputs
    assert len(outputs[Descriptors]) > 20


@pytest.mark.slow
def test_config_dict(pipeline_config):
    pipeline = Pipeline([Filename], [PipelineResult], **pipeline_config)
    config_dict = pipeline.get_config()
    print(config_dict)
    assert('Localizer' in config_dict)
    assert('Decoder' in config_dict)
    assert(config_dict['Localizer']['model_path'] == 'REQUIRED')
    assert(config_dict['Decoder']['model_path'] == 'REQUIRED')


@pytest.mark.slow
def test_no_detection(tmpdir, pipeline_config):
    repo = Repository(str(tmpdir))
    sink = BBBinaryRepoSink(repo, camId=0)

    pipeline = Pipeline([Image, Timestamp], [PipelineResult], **pipeline_config)

    image = np.zeros((3000, 4000), dtype=np.uint8)

    results = pipeline([image, 0])
    data_source = DataSource.new_message(filename='source')
    sink.add_frame(data_source, results, 0)
    sink.finish()

    assert(len(list(repo.iter_fnames())) == 1)

    for fname in repo.iter_fnames():
        with open(fname, 'rb') as f:
            fc = FrameContainer.read(f)
            assert(len(fc.frames) == 1)
            assert fc.dataSources[0].filename == 'source'

            frame = fc.frames[0]
            assert(len(frame.detectionsUnion.detectionsDP) == 0)


@pytest.mark.slow
def test_generator_processor(tmpdir, bees_image, pipeline_config):
    def image_generator():
        ts = time.time()
        data_source = DataSource.new_message(filename='bees.jpeg')
        for i in range(2):
            img = imread(bees_image)
            yield data_source, img, ts + i

    repo = Repository(str(tmpdir))
    pipeline = Pipeline([Image, Timestamp], [PipelineResult], **pipeline_config)
    gen_processor = GeneratorProcessor(
        pipeline, lambda: BBBinaryRepoSink(repo, camId=2))

    gen_processor(image_generator())
    gen_processor(image_generator())
    fnames = list(repo.iter_fnames())
    assert len(fnames) == 2

    last_ts = 0
    for fname in repo.iter_fnames():
        print("{}: {}".format(fname, os.path.getsize(fname)))
        with open(fname, 'rb') as f:
            fc = FrameContainer.read(f)
        assert fc.dataSources[0].filename == 'bees.jpeg'
        assert last_ts < fc.fromTimestamp
        last_ts = fc.fromTimestamp


@pytest.mark.slow
def test_generator_processor_video(tmpdir, bees_video, filelists_path, pipeline_config):
    repo = Repository(str(tmpdir))
    pipeline = Pipeline([Image, Timestamp], [PipelineResult], **pipeline_config)
    gen_processor = GeneratorProcessor(
        pipeline, lambda: BBBinaryRepoSink(repo, camId=0))

    gen = video_generator(bees_video, filelists_path)

    gen_processor(gen)
    fnames = list(repo.iter_fnames())
    assert len(fnames) == 1

    last_ts = 0
    num_frames = 0
    for fname in repo.iter_fnames():
        print("{}: {}".format(fname, os.path.getsize(fname)))
        with open(fname, 'rb') as f:
            fc = FrameContainer.read(f)
            num_frames += len(list(fc.frames))
        assert fc.dataSources[0].filename == os.path.basename(bees_video)
        assert last_ts < fc.fromTimestamp
        last_ts = fc.fromTimestamp

    assert(num_frames == 3)


@pytest.mark.slow
def test_generator_processor_threads(tmpdir, bees_video, filelists_path, pipeline_config):
    repo = Repository(str(tmpdir))
    pipelines = [Pipeline([Image, Timestamp], [PipelineResult], **pipeline_config) for
                 _ in range(3)]
    gen_processor = GeneratorProcessor(
        pipelines, lambda: BBBinaryRepoSink(repo, camId=0))

    gen = video_generator(bees_video, filelists_path)

    gen_processor(gen)
    fnames = list(repo.iter_fnames())
    assert len(fnames) == 1

    num_frames = 0
    for fname in repo.iter_fnames():
        with open(fname, 'rb') as f:
            fc = FrameContainer.read(f)
            num_frames += len(list(fc.frames))

    assert(num_frames == 3)


def test_crown_visualiser_on_a_bee(bee_in_the_center_image, outdir):
    bee_img = imread(bee_in_the_center_image) / 255.
    vis = ResultCrownVisualizer()
    bits = np.array([[1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]], dtype=np.float64)
    random = 0.45*np.random.random(bits.shape)
    bits[bits < 0.5] += random[bits < 0.5]
    bits[bits > 0.5] -= random[bits > 0.5]
    pos = np.array([(bee_img.shape[0] // 2, bee_img.shape[1] // 2)])
    z_angle = np.array([[np.radians(170), 0, 0]])
    overlay, = vis(bee_img, pos, z_angle, bits)
    img_with_overlay = ResultCrownVisualizer.add_overlay(bee_img, overlay)
    imsave(str(outdir.join("overlay.png")), overlay)
    imsave(str(outdir.join("overlay_0.png")), overlay[:, :, 0])
    imsave(str(outdir.join("overlay_1.png")), overlay[:, :, 1])
    imsave(str(outdir.join("overlay_2.png")), overlay[:, :, 2])
    imsave(str(outdir.join("overlay_3.png")), overlay[:, :, 3])
    imsave(str(outdir.join("crown.png")), img_with_overlay)


def test_localizer_visualizer(pipeline_results, bees_image, outdir):
    res = pipeline_results
    vis = LocalizerVisualizer(roi_overlay='circle')
    name, _ = os.path.splitext(os.path.basename(bees_image))
    overlay, = vis(res[Image], res[LocalizerPositions], {'roi_size': 100})
    imsave(str(outdir.join(name + "_localizer.png")), overlay)


def test_saliency_visualizer(pipeline_results, bees_image, outdir):
    res = pipeline_results
    vis = SaliencyVisualizer()
    name, _ = os.path.splitext(os.path.basename(bees_image))
    overlay, = vis(res[Image], res[SaliencyImage])
    imsave(str(outdir.join(name + "_saliencies.png")), overlay)


def test_crown_visualiser_on_a_image(pipeline_results, bees_image, outdir):
    vis = ResultCrownVisualizer()
    res = pipeline_results
    img = res[Image]
    overlay, = vis(res[Image], res[LocalizerPositions],
                   res[Orientations], res[IDs])
    overlay = zoom(overlay, (0.5, 0.5, 1), order=1)
    img = zoom(img, 0.5, order=3) / 255.
    img_with_overlay = ResultCrownVisualizer.add_overlay(img, overlay)

    name, _ = os.path.splitext(os.path.basename(bees_image))
    imsave(str(outdir.join(name + "_overlay.png")), overlay)
    imsave(str(outdir.join(name + "_added_overlay.jpeg")), img_with_overlay)


def test_auto_config():
    config = get_auto_config()
    assert 'Decoder' in config
    assert os.path.exists(config['Decoder']['model_path'])
    assert 'Localizer' in config
    assert os.path.exists(config['Localizer']['model_path'])
    assert 'TagSimilarityEncoder' in config
    assert os.path.exists(config['TagSimilarityEncoder']['model_path'])

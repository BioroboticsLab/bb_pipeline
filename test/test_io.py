from pipeline.io import unique_id, video_generator


def test_video_generator_2015(bees_video, filelists_path):
    gen = video_generator(bees_video, ts_format="2015", path_filelists=filelists_path)
    results = list(gen)
    assert len(results) == 3
    prev_ts = 0.0
    for _, _, ts in results:
        assert ts > prev_ts
        prev_ts = ts


def test_video_generator_2016(bees_video_2016):
    gen = video_generator(bees_video_2016, ts_format="2016", path_filelists=None)
    results = list(gen)
    assert len(results) == 4
    prev_ts = 0.0
    for _, _, ts in results:
        assert ts > prev_ts
        prev_ts = ts


def test_unique_id():
    first_id = unique_id()
    second_id = unique_id()
    assert first_id.bit_length() == 64
    assert second_id.bit_length() == 64
    assert first_id != second_id

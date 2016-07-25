from pipeline.io import video_generator, unique_id


def test_video_generator(bees_video, filelists_path):
    gen = video_generator(bees_video, filelists_path)
    results = list(gen)
    assert(len(results) == 3)
    prev_ts = 0.
    for _, _, ts in results:
        assert(ts > prev_ts)
        prev_ts = ts


def test_unique_id():
    first_id = unique_id()
    second_id = unique_id()
    assert(first_id.bit_length() == 64)
    assert(second_id.bit_length() == 64)
    assert(first_id != second_id)

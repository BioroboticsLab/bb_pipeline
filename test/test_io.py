from pipeline.io import video_generator


def test_video_generator(bees_video, filelists_path):
    gen = video_generator(bees_video, filelists_path)
    results = list(gen)
    assert(len(results) == 3)
    prev_ts = 0.
    for _, _, ts in results:
        assert(ts > prev_ts)
        prev_ts = ts

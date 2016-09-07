import os
import pytest

try:
    from mpi4py import MPI # noqa
    from pipeline.scripts.bb_pipeline_mpi import Mutex
    from pipeline.scripts.bb_pipeline_mpi import process_video as mpi_process_video
    run_mpi_test = True
except ImportError:
    run_mpi_test = False

from pipeline.scripts.bb_pipeline import process_video as cmdline_process_video
from bb_binary import Repository, FrameContainer


def check_repo(path, bees_video):
    repo = Repository(path)

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
@pytest.mark.skipif(not run_mpi_test, reason='mpi4py not installed')
def test_mpi_process_function(tmpdir, bees_video, filelists_path, pipeline_config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mutex = Mutex(comm)

    mpi_process_video(bees_video, filelists_path, str(tmpdir), rank, mutex)

    check_repo(str(tmpdir), bees_video)


@pytest.mark.slow
def test_process_function(tmpdir, bees_video, filelists_path, pipeline_config):
    tmpdir = str(tmpdir)

    class Args:
        num_threads = 1
        repo_output_path = tmpdir
        video_path = bees_video
        text_root_path = filelists_path

    args = Args()
    cmdline_process_video(args)

    check_repo(tmpdir, bees_video)

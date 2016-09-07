#!/usr/bin/env python3

import argparse
import sys
from mpi4py import MPI

from pipeline import Pipeline
from pipeline.cmdline import logger, get_shared_positional_arguments
from pipeline.pipeline import GeneratorProcessor, get_auto_config
from pipeline.io import LockedBBBinaryRepoSink, video_generator
from pipeline.objects import PipelineResult, Image, Timestamp
from bb_binary import Repository, parse_video_fname

from array import array as _array
import struct as _struct


# source: https://github.com/mpi4py/mpi4py/blob/master/demo/nxtval/nxtval-mpi3.py
class Counter:
    def __init__(self, comm):
        rank = comm.Get_rank()
        itemsize = MPI.INT.Get_size()
        if rank == 0:
            n = 1
        else:
            n = 0
        self.win = MPI.Win.Allocate(n*itemsize, itemsize,
                                    MPI.INFO_NULL, comm)
        if rank == 0:
            mem = self.win.memory
            mem[:] = _struct.pack('i', 0)

    def free(self):
        self.win.Free()

    def next(self, increment=1):
        incr = _array('i', [increment])
        nval = _array('i', [0])
        self.win.Lock(0)
        self.win.Get_accumulate([incr, 1, MPI.INT],
                                [nval, 1, MPI.INT],
                                0, op=MPI.SUM)
        self.win.Unlock(0)
        return nval[0]


# source: https://github.com/mpi4py/mpi4py/blob/master/demo/nxtval/nxtval-mpi3.py
class Mutex:
    def __init__(self, comm):
        self.counter = Counter(comm)

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, *exc):
        self.unlock()
        return None

    def free(self):
        self.counter.free()

    def lock(self):
        value = self.counter.next(+1)
        while value != 0:
            value = self.counter.next(-1)
            value = self.counter.next(+1)

    def unlock(self):
        self.counter.next(-1)


def process_video(video_path, text_root_path, repo_output_path, rank, mutex):
    info = lambda msg: logger.info('Process {}: {}'.format(rank, msg))
    config = get_auto_config()

    info('Initializing pipeline')
    pipeline = Pipeline([Image, Timestamp], [PipelineResult], **config)

    info('Loading bb_binary repository {}'.format(repo_output_path))
    repo = Repository(repo_output_path)

    camId, _, _ = parse_video_fname(video_path)
    info('Parsed camId = {}'.format(camId))
    gen_processor = GeneratorProcessor(pipeline,
                                       lambda: LockedBBBinaryRepoSink(repo,
                                                                      camId=camId,
                                                                      mutex=mutex))

    log_callback = lambda frame_idx: info('Processing frame {} from {}'.format(frame_idx,
                                                                               video_path))

    info('Processing video frames from {}'.format(video_path))
    gen_processor(video_generator(video_path, text_root_path, log_callback))


def parse_args(comm):
    parser = argparse.ArgumentParser(
        prog='BeesBook MPI batch processor',
        description='Batch process video using the beesbook pipeline')

    parser.add_argument('video_list_path',
                        help='root path of input video list',
                        type=str)

    for arg, kwargs in get_shared_positional_arguments():
        parser.add_argument(arg, **kwargs)

    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = comm.bcast(args, root=0)

    args = (args.video_list_path, args.text_root_path, args.repo_output_path)

    if any([a is None for a in args]):
        if comm.Get_rank() == 0:
            print(parser.print_help())
        exit(1)

    return args


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mutex = Mutex(comm)

    video_list_path, text_root_path, repo_output_path = parse_args(comm)

    video_paths = [s.strip() for s in open(video_list_path, 'r').readlines()]

    abort = False
    if rank is 0:
        logger.info('There are {} processes'.format(comm.Get_size()))

        if (len(video_paths)) != comm.Get_size():
            logger.error(('Process was started with {} processes, but `{}` contains ' +
                          '{} paths. Aborting.').format(comm.Get_size(),
                                                        video_list_path,
                                                        len(video_paths)))
            sys.stderr.flush()
            abort = True

    abort = comm.bcast(abort, root=0)
    comm.Barrier()

    if not abort:
        process_video(video_paths[rank],
                      text_root_path,
                      repo_output_path,
                      rank,
                      mutex)

    comm.Barrier()


if __name__ == '__main__':
    main()

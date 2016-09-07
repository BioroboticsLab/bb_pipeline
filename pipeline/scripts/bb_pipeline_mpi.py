#!/usr/bin/env python3

import argparse
import sys
from mpi4py import MPI

from pipeline import Pipeline
from pipeline.cmdline import logger, get_shared_positional_arguments
from pipeline.pipeline import GeneratorProcessor, get_auto_config
from pipeline.io import BBBinaryRepoSink, video_generator
from pipeline.objects import PipelineResult, Image, Timestamp
from bb_binary import Repository, parse_video_fname


def process_video(video_path, text_root_path, repo_output_path, rank):
    info = lambda msg: logger.info('Process {}: {}'.format(rank, msg))
    config = get_auto_config()

    info('Initializing pipeline')
    pipeline = Pipeline([Image, Timestamp], [PipelineResult], **config)

    info('Loading bb_binary repository {}'.format(repo_output_path))
    repo = Repository(repo_output_path)

    camId, _, _ = parse_video_fname(video_path)
    info('Parsed camId = {}'.format(camId))
    gen_processor = GeneratorProcessor(pipeline, lambda: BBBinaryRepoSink(repo, camId=camId))

    info('Processing video frames from {}'.format(video_path))
    gen_processor(video_generator(video_path, text_root_path))


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


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
                      rank)

    comm.Barrier()

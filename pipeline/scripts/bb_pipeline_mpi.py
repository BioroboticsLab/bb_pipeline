#!/usr/bin/env python3

import argparse
import atexit
import os
import shutil
import sys
from mpi4py import MPI


def process_video(video_path, text_root_path, repo_output_path, rank):
    info = lambda msg: logger.info('Process {}: {}'.format(rank, msg))

    import theano
    from pipeline import Pipeline
    from pipeline.cmdline import logger
    from pipeline.pipeline import GeneratorProcessor, get_auto_config
    from pipeline.io import BBBinaryRepoSink, video_generator
    from pipeline.objects import PipelineResult, Image, Timestamp
    from bb_binary import Repository, parse_video_fname

    repo_output_path = os.path.join(repo_output_path, 'process_{}'.format(rank))

    info('Theano compile dir: {}'.format(theano.config.base_compiledir))
    info('Output dir: {}'.format(repo_output_path))

    config = get_auto_config()

    info('Initializing pipeline')
    pipeline = Pipeline([Image, Timestamp], [PipelineResult], **config)

    info('Loading bb_binary repository {}'.format(repo_output_path))
    repo = Repository(repo_output_path)

    camId, _, _ = parse_video_fname(video_path)
    info('Parsed camId = {}'.format(camId))
    gen_processor = GeneratorProcessor(pipeline,
                                       lambda: BBBinaryRepoSink(repo, camId=camId))

    log_callback = lambda frame_idx: info('Processing frame {} from {}'.format(frame_idx,
                                                                               video_path))
    ffmpeg_stderr_fd = open('process_{}_ffmpeg_stderr.log'.format(rank), 'w')

    info('Processing video frames from {}'.format(video_path))
    gen_processor(video_generator(video_path, text_root_path, log_callback, ffmpeg_stderr_fd))


def parse_args(comm):
    from pipeline.cmdline import get_shared_positional_arguments

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


def delete_folder(path):
    shutil.rmtree(path)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if 'PBS_O_WORKDIR' in os.environ:
        compile_dir = '{}/theano_compile_process_{}'.format(os.environ['PBS_O_WORKDIR'], rank)
        os.environ["THEANO_FLAGS"] = ("base_compiledir='{}'".format(compile_dir))

        atexit.register(delete_folder, compile_dir)

    from pipeline.cmdline import logger
    info = lambda msg: logger.info('Process {}: {}'.format(rank, msg))

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

    if not abort:
        process_video(video_paths[rank],
                      text_root_path,
                      repo_output_path,
                      rank)

    info('Exiting.')


if __name__ == '__main__':
    main()

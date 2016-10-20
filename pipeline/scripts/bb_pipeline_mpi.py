#!/usr/bin/env python3

import argparse
import atexit
from itertools import chain
import os
import shutil
from mpi4py import MPI


def process_video(video_path, repo_output_path, ts_format, text_root_path, rank):
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
    gen_processor(video_generator(video_path, ts_format, text_root_path,
                                  log_callback, ffmpeg_stderr_fd))


def parse_args(comm):
    from pipeline.cmdline import get_shared_positional_arguments, get_shared_optional_arguments

    parser = argparse.ArgumentParser(
        prog='BeesBook MPI batch processor',
        description='Batch process video using the beesbook pipeline')

    parser.add_argument('video_list_path',
                        help='root path of input video list',
                        type=str)

    for arg, kwargs in chain(get_shared_positional_arguments(), get_shared_optional_arguments()):
        parser.add_argument(arg, **kwargs)

    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = comm.bcast(args, root=0)

    parsed_args = [args.video_list_path, args.repo_output_path, args.timestamp_format]

    if any([a is None for a in parsed_args]):
        if comm.Get_rank() == 0:
            print(parser.print_help())
        exit(1)

    # can be None
    parsed_args.append(args.text_root_path)

    return parsed_args


def delete_folder(path):
    shutil.rmtree(path)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if 'PIPELINE_TMP_DIR' in os.environ:
        compile_dir = '{}/theano_compile_process_{}'.format(os.environ['PIPELINE_TMP_DIR'], rank)
        os.environ["THEANO_FLAGS"] = ("base_compiledir='{}'".format(compile_dir))

        atexit.register(delete_folder, compile_dir)

    from pipeline.cmdline import logger
    info = lambda msg: logger.info('Process {}: {}'.format(rank, msg))

    video_list_path, repo_output_path, ts_format, text_root_path = parse_args(comm)

    video_paths = [s.strip() for s in open(video_list_path, 'r').readlines()]

    if rank is 0:
        logger.info('There are {} processes'.format(comm.Get_size()))

    if rank < len(video_paths):
        process_video(video_paths[rank],
                      repo_output_path,
                      ts_format,
                      text_root_path,
                      rank)
    else:
        logger.warning('Process {}: No file to process'.format(rank))

    info('Reached Barrier.')
    comm.Barrier()
    info('Exiting.')


if __name__ == '__main__':
    main()

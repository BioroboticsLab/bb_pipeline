#!/usr/bin/env python3

import argparse
import atexit
import os
import shutil
from itertools import chain

from mpi4py import MPI


def process_video(video_path, repo_output_path, ts_format, text_root_path, rank):
    info = lambda msg: logger.info(f"Process {rank}: {msg}")

    import theano
    from pipeline import Pipeline
    from pipeline.cmdline import logger
    from pipeline.pipeline import GeneratorProcessor, get_auto_config
    from pipeline.io import BBBinaryRepoSink, video_generator
    from pipeline.objects import PipelineResult, Image, Timestamp
    from bb_binary import Repository, parse_video_fname

    repo_output_path = os.path.join(repo_output_path, f"process_{rank}")

    info(f"Theano compile dir: {theano.config.base_compiledir}")
    info(f"Output dir: {repo_output_path}")

    config = get_auto_config()

    info("Initializing pipeline")
    pipeline = Pipeline([Image, Timestamp], [PipelineResult], **config)

    info(f"Loading bb_binary repository {repo_output_path}")
    repo = Repository(repo_output_path)

    camId, _, _ = parse_video_fname(video_path)
    info(f"Parsed camId = {camId}")
    gen_processor = GeneratorProcessor(
        pipeline, lambda: BBBinaryRepoSink(repo, camId=camId)
    )

    log_callback = lambda frame_idx: info(
        f"Processing frame {frame_idx} from {video_path}"
    )
    ffmpeg_stderr_fd = open(f"process_{rank}_ffmpeg_stderr.log", "w")

    info(f"Processing video frames from {video_path}")
    gen_processor(
        video_generator(
            video_path, ts_format, text_root_path, log_callback, ffmpeg_stderr_fd
        )
    )


def parse_args(comm):  # pragma: no cover
    from pipeline.cmdline import (
        get_shared_positional_arguments,
        get_shared_optional_arguments,
    )

    parser = argparse.ArgumentParser(
        prog="BeesBook MPI batch processor",
        description="Batch process video using the beesbook pipeline",
    )

    parser.add_argument(
        "video_list_path", help="root path of input video list", type=str
    )

    for arg, kwargs in chain(
        get_shared_positional_arguments(), get_shared_optional_arguments()
    ):
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


def delete_folder(path):  # pragma: no cover
    shutil.rmtree(path)


def main():  # pragma: no cover
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if "PIPELINE_TMP_DIR" in os.environ:
        compile_dir = "{}/theano_compile_process_{}".format(
            os.environ["PIPELINE_TMP_DIR"], rank
        )
        os.environ["THEANO_FLAGS"] = f"base_compiledir='{compile_dir}'"

        atexit.register(delete_folder, compile_dir)

    from pipeline.cmdline import logger

    info = lambda msg: logger.info(f"Process {rank}: {msg}")

    video_list_path, repo_output_path, ts_format, text_root_path = parse_args(comm)

    video_paths = [s.strip() for s in open(video_list_path).readlines()]

    if rank == 0:
        logger.info(f"There are {comm.Get_size()} processes")

    if rank < len(video_paths):
        process_video(
            video_paths[rank], repo_output_path, ts_format, text_root_path, rank
        )
    else:
        logger.warning(f"Process {rank}: No file to process")

    info("Reached Barrier.")
    comm.Barrier()
    info("Exiting.")


if __name__ == "__main__":  # pragma: no cover
    main()

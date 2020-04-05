#!/usr/bin/env python3

import argparse
from itertools import chain

from pipeline import Pipeline
from pipeline.cmdline import (
    get_shared_optional_arguments,
    get_shared_positional_arguments,
    logger,
)
from pipeline.io import BBBinaryRepoSink, video_generator
from pipeline.objects import Image, PipelineResult, Timestamp
from pipeline.pipeline import GeneratorProcessor, get_auto_config

from bb_binary import Repository, parse_video_fname


def process_video(args):
    config = get_auto_config()

    logger.info(f"Initializing {args.num_threads} pipeline(s)")
    plines = [
        Pipeline([Image, Timestamp], [PipelineResult], **config)
        for _ in range(args.num_threads)
    ]

    logger.info(f"Loading bb_binary repository {args.repo_output_path}")
    repo = Repository(args.repo_output_path)

    camId, _, _ = parse_video_fname(args.video_path)
    logger.info(f"Parsed camId = {camId}")
    gen_processor = GeneratorProcessor(
        plines, lambda: BBBinaryRepoSink(repo, camId=camId), use_tqdm=args.progressbar
    )

    logger.info(f"Processing video frames from {args.video_path}")
    gen_processor(
        video_generator(args.video_path, args.timestamp_format, args.text_root_path)
    )


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="BeesBook pipeline",
        description="Process a video using the beesbook pipeline",
    )

    parser.add_argument("video_path", help="path of input video", type=str)
    for arg, kwargs in chain(
        get_shared_positional_arguments(), get_shared_optional_arguments()
    ):
        parser.add_argument(arg, **kwargs)

    args = parser.parse_args()

    logger.info(f"Processing video: {args.video_path}")
    logger.info(f"Config: {args}")

    process_video(args)


if __name__ == "__main__":  # pragma: no cover
    main()

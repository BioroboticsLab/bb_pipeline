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
    """
    Processes a video using the BeesBook pipeline with specified configurations.

    Args:
        args (Namespace): Arguments with required and optional attributes for processing the video.

    Required Attributes:
        video_path (str): Path to the input video file to be processed.
        repo_output_path (str): Path to the output directory for the bb_binary repository.
        num_threads (int): Number of pipeline threads to use.
        timestamp_format (str): Format of the timestamps in the video frames.
        text_root_path (str): Root path for any text-based metadata associated with the video.

    Optional Attributes (handled by `getattr`):
        decoder_model_path (str, optional): Path to the decoder model (e.g., `decoder_2019_keras3.h5`).
        localizer_model_path (str, optional): Path to the localizer model (e.g., `localizer_2019_keras3.h5`).
        localizer_attributes_path (str, optional): Path to the localizer attributes JSON file (e.g., `localizer_2019_attributes.json`).
        video_file_type (str, optional): Format type for the video file; defaults to 'auto'.
        progressbar (bool, optional): If `True`, enables a progress bar display for processing.
    
    """
    config = get_auto_config()

    # Use getattr to handle cases where optional arguments might not be provided in args
    config['Decoder']['model_path'] = getattr(args, 'decoder_model_path', config['Decoder']['model_path'])
    config['Localizer']['model_path'] = getattr(args, 'localizer_model_path', config['Localizer']['model_path'])
    config['Localizer']['attributes_path'] = getattr(args, 'localizer_attributes_path', config['Localizer']['attributes_path'])

    logger.info(f"Initializing {args.num_threads} pipeline(s)")
    plines = [
        Pipeline([Image, Timestamp], [PipelineResult], **config)
        for _ in range(args.num_threads)
    ]

    logger.info(f"Loading bb_binary repository {args.repo_output_path}")
    repo = Repository(args.repo_output_path)

    # Set default value for video_file_type if it doesn't exist in args
    video_file_type = getattr(args, 'video_file_type', 'auto')
    camId, _, _ = parse_video_fname(args.video_path, format=video_file_type)

    logger.info(f"Parsed camId = {camId}")
    gen_processor = GeneratorProcessor(
        plines, lambda: BBBinaryRepoSink(repo, camId=camId), use_tqdm=getattr(args, 'progressbar', False)
    )

    logger.info(f"Processing video frames from {args.video_path}")
    gen_processor(
        video_generator(args.video_path, getattr(args, 'timestamp_format', None), getattr(args, 'text_root_path', None))
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

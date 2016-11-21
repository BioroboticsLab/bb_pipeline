#!/usr/bin/env python3

import argparse
from itertools import chain

from pipeline import Pipeline
from pipeline.cmdline import logger, get_shared_positional_arguments, get_shared_optional_arguments
from pipeline.pipeline import GeneratorProcessor, get_auto_config
from pipeline.io import BBBinaryRepoSink, video_generator
from pipeline.objects import PipelineResult, Image, Timestamp
from bb_binary import Repository, parse_video_fname


def process_video(args):
    config = get_auto_config()
    
    config['TagSimilarityEncoder']['model_path'] = args.encoder_model
    logger.info('Use encoder model in path {}'.format(args.encoder_model))

    logger.info('Initializing {} pipeline(s)'.format(args.num_threads))
    plines = [Pipeline([Image, Timestamp], [PipelineResult], **config)
              for _ in range(args.num_threads)]

    logger.info('Loading bb_binary repository {}'.format(args.repo_output_path))
    repo = Repository(args.repo_output_path)

    camId, _, _ = parse_video_fname(args.video_path)
    logger.info('Parsed camId = {}'.format(camId))
    gen_processor = GeneratorProcessor(plines, lambda: BBBinaryRepoSink(repo, camId=camId))

    logger.info('Processing video frames from {}'.format(args.video_path))
    gen_processor(video_generator(args.video_path, args.text_root_path))


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook pipeline',
        description='Process a video using the beesbook pipeline')

    parser.add_argument('video_path', help='path of input video', type=str)
    for arg, kwargs in chain(get_shared_positional_arguments(), get_shared_optional_arguments()):
        parser.add_argument(arg, **kwargs)

    args = parser.parse_args()

    logger.info('Processing video: {}'.format(args.video_path))
    logger.info('Config: {}'.format(args))

    process_video(args)


if __name__ == '__main__':
    main()

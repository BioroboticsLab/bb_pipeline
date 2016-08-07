#!/usr/bin/env python3

import argparse
import os
import shutil
from itertools import chain
from subprocess import call

from pipeline.cmdline import logger, get_shared_positional_arguments, get_shared_optional_arguments


def main():
    parser = argparse.ArgumentParser(
        prog='BeesBook pipeline batch processor',
        description='Batch process video using the beesbook pipeline')

    parser.add_argument('video_root_path', help='root path of input videos', type=str)
    for arg, kwargs in chain(get_shared_positional_arguments(), get_shared_optional_arguments()):
        parser.add_argument(arg, **kwargs)

    args = parser.parse_args()

    pipeline_cmd = None
    if os.path.exists(os.path.join(os.getcwd(), 'bb_pipeline')):
        pipeline_cmd = os.path.join(os.getcwd(), 'bb_pipeline')
    else:
        pipeline_cmd = shutil.which('bb_pipeline')
    assert(pipeline_cmd is not None)

    video_files = []
    for root, dirs, files in os.walk(args.video_root_path):
        for file in files:
            video_files.append(os.path.join(root, file))

    logger.info('Processing files: \n\t{}'.format('\n\t'.join(video_files)))

    for fname in video_files:
        cmd_args = [fname] + \
            list([args.__dict__[param] for param, _ in get_shared_positional_arguments()])
        for param, _ in get_shared_optional_arguments():
            if param[2:] in args.__dict__:
                cmd_args.append(param)
                cmd_args.append(str(args.__dict__[param[2:]]))
        call([pipeline_cmd] + cmd_args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
import os
import pathlib
import shutil
from itertools import chain
from subprocess import call

from ..cmdline import (
    get_shared_optional_arguments,
    get_shared_positional_arguments,
    logger,
)

from .bb_pipeline import process_video


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="BeesBook pipeline batch processor",
        description="Batch process video using the beesbook pipeline",
    )

    parser.add_argument(
        "--video_glob_pattern",
        help="glob pattern for files to process",
        type=str,
        default=None,
    )
    parser.add_argument("video_root_path", help="root path of input videos", type=str)
    for arg, kwargs in chain(
        get_shared_positional_arguments(), get_shared_optional_arguments()
    ):
        parser.add_argument(arg, **kwargs)

    args = parser.parse_args()

    pipeline_cmd = None
    if os.path.exists(os.path.join(os.getcwd(), "bb_pipeline")):
        pipeline_cmd = os.path.join(os.getcwd(), "bb_pipeline")
    else:
        pipeline_cmd = shutil.which("bb_pipeline")
    assert pipeline_cmd is not None

    video_files = []
    for root, dirs, files in os.walk(args.video_root_path):
        for file in files:
            # Ignore timestamp files.
            if not file.endswith("txt"):
                full_path = os.path.join(root, file)
                # Ignore files that do not match the given pattern.
                if args.video_glob_pattern and args.video_glob_pattern != "None":
                    if not pathlib.PurePath(full_path).match(args.video_glob_pattern):
                        continue
                video_files.append(full_path)

    logger.info(
        "Processing {} files: \n\t{}".format(len(video_files), "\n\t".join(video_files))
    )

    for fname in video_files:
        args.video_path = fname
        process_video(args)


if __name__ == "__main__":
    main()

import logging


def setup_logging():
    logger = logging.getLogger('beesbook_pipeline')

    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

logger = setup_logging()


def get_shared_positional_arguments():
    arguments = list()
    arguments.append(('text_root_path', {'help': 'root path for beesbook image name text files',
                                         'type': str}))
    arguments.append(('repo_output_path', {'help': 'root path for bb_binary output repository',
                                           'type': str}))
    return arguments


def get_shared_optional_arguments():
    arguments = list()
    arguments.append(('--num_threads', {'help': 'number of images to process in parallel',
                                        'type': int, 'default': 1}))
    arguments.append(('--encoder_model', {'help': 'path to TagSimilarityEncoder model', 'type': str, 'default': None}))
    return arguments

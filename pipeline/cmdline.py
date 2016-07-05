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
    arguments.append(('saliency_weight_path', {'help': 'path of saliency model weights',
                                               'type': str}))
    arguments.append(('decoder_weight_path', {'help': 'path of decoder model weights',
                                              'type': str}))
    arguments.append(('decoder_model_path', {'help': 'path of decoder model architecture',
                                             'type': str}))
    return arguments


def get_shared_optional_arguments():
    arguments = list()
    arguments.append(('--saliency_threshold', {'help': 'threshold for saliency localizer',
                                               'type': float, 'default': 0.1}))
    arguments.append(('--num_threads', {'help': 'number of images to process in parallel',
                                        'type': int, 'default': 1}))
    return arguments

#!/usr/bin/env python3

from tempfile import NamedTemporaryFile
import json
from threading import Lock

import numpy as np
from flask import Flask, request
from scipy.misc import imread
from pipeline import Pipeline
from pipeline.objects import Image, LocalizerPositions, Saliencies, IDs
from pipeline.pipeline import get_auto_config

app = Flask(__name__)


def init_pipeline():
    pipeline = Pipeline([Image],
                        [LocalizerPositions, Saliencies, IDs],
                        **get_auto_config())
    return pipeline

pipeline = init_pipeline()
pipeline_lock = Lock()


def jsonify(instance):
    if isinstance(instance, np.ndarray):
        return instance.tolist()
    return instance


def process_image(image):
    with pipeline_lock:
        results = pipeline([image])
    return json.dumps(dict([(k.__name__, jsonify(v)) for k, v in
                            results.items()]), ensure_ascii=False)


@app.route('/process', methods=['POST'])
def api_message():
    print('Retrieving process request')
    if request.headers['Content-Type'] == 'application/octet-stream':
        try:
            with NamedTemporaryFile(delete=True) as f:
                f.write(request.data)
                image = imread(f)
                return process_image(image)
        except Exception as err:
            return '{}'.format(err)
    else:
        return "415 Unsupported Media Type"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

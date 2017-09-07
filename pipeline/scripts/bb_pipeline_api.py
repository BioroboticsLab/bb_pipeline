#!/usr/bin/env python3
"""This script provides a RESTful remote endpoint to the detection pipeline.
An image is sent to the server, which sends back the requested results.
"""

from tempfile import NamedTemporaryFile
import json
from urllib import parse
import inspect

import numpy as np
from flask import Flask, request
from scipy.misc import imread, imsave
import msgpack
import io
import cachetools
from pipeline import Pipeline
from pipeline import objects
from pipeline.pipeline import get_auto_config

app = Flask(__name__)

default_output = [objects.LocalizerPositions, objects.Saliencies, objects.IDs]
pipeline_cache = cachetools.LRUCache(maxsize=4)


def init_pipeline(output):
    """Helper function to initialize a new pipeline
    that generates the desired output.

    Args:
        output (list): pipeline objects that the pipeline will
                       generate as the result

    Returns:
        pipeline object
    """
    pipeline = Pipeline([objects.Image],
                        output,
                        **get_auto_config())
    return pipeline


def get_cached_pipeline(output):
    """Helper function to get a pipeline that generates the desired output.
    Pipelines are stored in a 'least recently used'-cache of size 4.
    If a pipeline with the specified output is already present in the
    cache, it will be used, otherwise a new one will be created
    and stored in the cache.

    Args:
        output (list): pipeline objects that the pipeline will
                       generate as the result

    Returns:
        pipeline object
    """
    if not (output):
        output_key = frozenset(default_output)
    else:
        output_key = frozenset(output)
    if output_key in pipeline_cache:
        print('Pipeline is cached.')
        return pipeline_cache[output_key]
    else:
        print('Pipeline is not cached, initializing new pipeline...')
        pipeline = init_pipeline(output)
        print('...done. Adding to cache.')
        pipeline_cache[output_key] = pipeline
        return pipeline


def png_encode(instance):
    """Helper function to convert a numpy array to a PNG image.

    Args:
        instance (ndarray): numpy array containing
                            the image to be converted

    Returns:
        bytes: bytes containing the PNG-encoded image.
    """
    if isinstance(instance, np.ndarray):
        b = io.BytesIO()
        imsave(b, instance, 'png')
        return b.getvalue()
    return instance


def process_image(pipeline, image, png):
    """Helper function to execute a pipeline and get the results.

    Args:
        pipeline: pipeline to be executed
        image (ndarray): image as input to the pipeline
        png (list of str): names of the outputs that will be converted to PNG

    Returns:
        msgpack: dictionary (serialized as a msgpack) containing
                 the results of the pipeline with the object names as keys
    """
    pipeline_results = pipeline([image])
    results_dict = {}
    for (k, v) in pipeline_results.items():
        results_dict[k.__name__] = (png_encode(
            v) if (k.__name__ in png) else v.tolist())
    return msgpack.packb(results_dict)


@app.route('/process', methods=['POST'])
def api_message():
    """This function handles the `/process` URL call.
    An image is appended as data to the request and the result is returned.

    The desired output objects can be specified by assigning a JSON list
    to the optional `output` URL parameter (remember to percent-encode
    the string first). If omitted, the default output will be returned
    ('LocalizerPositions', 'Saliencies', 'IDs').

    Additionally, any output that should be encoded as PNG (image output,
    e.g. 'CrownOverlay') can be assigned to the optional `png` URL parameter,
    also as a JSON list.


    Example:

    .. code::

        import requests
        import json
        from urllib import parse
        import io
        import msgpack
        from scipy.misc import imread

        with open('/local/image/file.png', 'rb') as image_file:
            headers = {'Content-type': 'application/octet-stream'}
            output_json = parse.quote(json.dumps(
                    ['LocalizerPositions', 'IDs', 'CrownOverlay']))
            png_json = parse.quote(json.dumps(
                    ['CrownOverlay']))
            url_params = {'output': output_json, 'png': png_json}
            url = 'http://localhost:10000/process'
            result = requests.post(
                url,
                params=url_params,
                data=image_file.read(),
                headers=headers)

    The serialized response will be stored in `result`.
    The results can be loaded like this:

    .. code::

        # Deserialize results
        result_unpacked = msgpack.unpackb(result.content)
        ids_list = result_unpacked[b'IDs']
        crownoverlay_image = imread(io.BytesIO(result_unpacked[b'CrownOverlay']))

    Note:
        The keys in the result dict are binary, since msgpack does not
        support string keys.
    """
    print('Retrieving process request')
    url_output_param = request.args.get('output')
    if url_output_param is None:
        print('No output specified, using defaults')
        pipeline = get_cached_pipeline(default_output)
    else:
        output_strings = json.loads(parse.unquote(url_output_param))
        if not output_strings:
            print('No output specified, using defaults')
            pipeline = get_cached_pipeline(default_output)
        else:
            output_objects = []
            for o in output_strings:
                if o in ['PipelineObject', 'PipelineObjectDescription',
                         'NumpyArrayDescription', 'FilenameDescription']:
                    print('Error: Illegal pipeline output specified: {}'.format(o))
                    return 'Error: Illegal pipeline output specified: {}'.format(o)
                else:
                    c = objects.__dict__.get(o)
                    if (c is None) or (not inspect.isclass(c)):
                        print('Invalid pipeline output specified: {}'.format(o))
                        return 'Invalid pipeline output specified: {}'.format(o)
                    else:
                        output_objects.append(c)
            print('Specified output: {}'.format(
                [o.__name__ for o in output_objects]))
            output = frozenset(output_objects)
            pipeline = get_cached_pipeline(output)
    png_please = request.args.get('png')
    if png_please is None:
        png = []
    else:
        png = json.loads(parse.unquote(png_please))
        if png:
            print('Specified png-encoded output: {}'.format([p for p in png]))
    if request.headers['Content-Type'] == 'application/octet-stream':
        try:
            with NamedTemporaryFile(delete=True) as f:
                print('Loading image...')
                f.write(request.data)
                image = imread(f)
                print('Processing image...')
                return process_image(pipeline, image, png)
        except Exception as err:
            return '{}'.format(err)
    else:
        return '415 Unsupported Media Type'


def main():  # pragma: no cover
    app.run(host='0.0.0.0', port=10000)


if __name__ == '__main__':  # pragma: no cover
    main()

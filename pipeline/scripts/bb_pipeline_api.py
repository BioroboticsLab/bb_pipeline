#!/usr/bin/env python3
"""This script provides a RESTful remote endpoint to the detection pipeline.
An image is sent to the server, which sends back the requested results.
"""

import inspect
import io
import json
from tempfile import NamedTemporaryFile
from urllib import parse

import cachetools
import msgpack
import numpy as np
from flask import Flask, abort, request
from scipy.misc import imread, imsave

from pipeline import Pipeline, objects
from pipeline.pipeline import get_auto_config

app = Flask(__name__)

default_output = [objects.LocalizerPositions, objects.Saliencies, objects.IDs]

pipeline_cache = cachetools.LRUCache(maxsize=4)
no_localizer_pipeline_cache = cachetools.LRUCache(maxsize=4)


def init_pipeline(output, no_localizer):
    """Helper function to initialize a new pipeline
    that generates the desired output.

    Args:
        output (list): pipeline objects that the pipeline will
                       generate as the result
        no_localizer (boolean): whether or not the localizer should be
                                skipped to decode a single tag in the
                                center of a 100x100 image

    Returns:
        pipeline object
    """

    if no_localizer:
        pipeline = Pipeline(
            [objects.Regions, objects.LocalizerPositions], output, **get_auto_config()
        )
    else:
        pipeline = Pipeline([objects.Image], output, **get_auto_config())
    return pipeline


def get_cached_pipeline(output, no_localizer):
    """Helper function to get a pipeline that generates the desired output.
    Pipelines are stored in a 'least recently used'-cache of size 4.
    If a pipeline with the specified output is already present in the
    cache, it will be used, otherwise a new one will be created
    and stored in the cache.

    Args:
        output (list): pipeline objects that the pipeline will
                       generate as the result
        no_localizer (boolean): determines if a pipeline should be
                                returned that skips the localizer to
                                decode a single tag in the center of a
                                100x100 image

    Returns:
        pipeline object
    """
    if not (output):
        output_key = frozenset(default_output)
    else:
        output_key = frozenset(output)
    if no_localizer:
        cache = no_localizer_pipeline_cache
    else:
        cache = pipeline_cache
    if output_key in cache:
        print("Pipeline is cached.")
        return cache[output_key]
    else:
        print("Pipeline is not cached, initializing new pipeline...")
        pipeline = init_pipeline(output, no_localizer)
        print("...done. Adding to cache.")
        cache[output_key] = pipeline
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
        imsave(b, instance, "png")
        return b.getvalue()
    return instance


def process_image(pipeline, image, png, no_localizer):
    """Helper function to execute a pipeline and get the results.

    Args:
        pipeline: pipeline to be executed
        image (ndarray): image as input to the pipeline
        png (list of str): names of the outputs that will be converted to PNG
        no_localizer (boolean): whether or not the localizer should be
                                skipped to decode a single tag in the
                                center of a 100x100 image

    Returns:
        msgpack: dictionary (serialized as a msgpack) containing
                 the results of the pipeline with the object names as keys
    """
    if no_localizer:
        positions = np.zeros((1, 2))
        regions = image[np.newaxis, np.newaxis, :, :]
        pipeline_results = pipeline([regions, positions])
    else:
        pipeline_results = pipeline([image])
    results_dict = {}
    for (k, v) in pipeline_results.items():
        results_dict[k.__name__] = png_encode(v) if (k.__name__ in png) else v.tolist()
    return msgpack.packb(results_dict)


@app.route("/decode/<mode>", methods=["POST"])
def api_message(mode):
    """This function handles the `/decode` URL call.
    The next URL segment determines the decoding mode, `/single` for the
    decoding of a single tag in the center of a 100x100 image and
    `/automatic` for the localization of tags in an image.

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
            url = 'http://localhost:10000/decode/automatic'
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
    if mode not in ["single", "automatic"]:
        abort(404)
    if request.headers["Content-Type"] != "application/octet-stream":
        abort(415)

    no_localizer = mode == "single"

    print("\nRetrieving process request")
    print("Decoding mode: {}".format("single" if no_localizer else "automatic"))
    url_output_param = request.args.get("output")
    if url_output_param is None:
        print("No output specified, using defaults")
        pipeline = get_cached_pipeline(default_output, no_localizer)
    else:
        output_strings = json.loads(parse.unquote(url_output_param))
        if not output_strings:
            print("No output specified, using defaults")
            pipeline = get_cached_pipeline(default_output, no_localizer)
        else:
            output_objects = []
            for o in output_strings:
                if o in [
                    "PipelineObject",
                    "PipelineObjectDescription",
                    "NumpyArrayDescription",
                    "FilenameDescription",
                ]:
                    print(f"Illegal pipeline output specified: {o}")
                    return f"Illegal pipeline output specified: {o}"
                else:
                    c = objects.__dict__.get(o)
                    if (c is None) or (not inspect.isclass(c)):
                        print(f"Invalid pipeline output specified: {o}")
                        return f"Invalid pipeline output specified: {o}"
                    else:
                        output_objects.append(c)
            print("Specified output: {}".format([o.__name__ for o in output_objects]))
            output = frozenset(output_objects)
            pipeline = get_cached_pipeline(output, no_localizer)
    png_please = request.args.get("png")
    if png_please is None:
        png = []
    else:
        png = json.loads(parse.unquote(png_please))
        if png:
            print("Specified png-encoded output: {}".format([p for p in png]))
    try:
        with NamedTemporaryFile(delete=True) as f:
            print("Loading image...")
            f.write(request.data)
            image = imread(f)
            if no_localizer and (image.shape != (100, 100)):
                print(f"Input image has wrong dimensions: {image.shape}")
                return f"Input image has wrong dimensions: {image.shape}"
            print("Processing image...")
            return process_image(pipeline, image, png, no_localizer)
    except Exception as err:
        print(f"Exception: {err}")
        return f"Exception: {err}"


def main():  # pragma: no cover
    app.run(host="0.0.0.0", port=10000)


if __name__ == "__main__":  # pragma: no cover
    main()

from joblib import Parallel, delayed
import pipeline.stages
from collections import defaultdict
import inspect
import urllib
import configparser
from inspect import Parameter
import os
import copy

# for backward compatibility
from pipeline.io import video_generator, BBBinaryRepoSink  # noqa


def _processSingleInput(pipeline, data_source, img, ts):
    return data_source, pipeline([img, ts]), ts


class GeneratorProcessor(object):
    def __init__(self, pipelines, sink_factory):
        if type(pipelines) == Pipeline:
            pipelines = [pipelines]
        self.pipelines = pipelines
        self.parallel = Parallel(n_jobs=len(pipelines), backend='threading')
        self.sink_factory = sink_factory

    def __call__(self, generator):
        sink = self.sink_factory()
        evaluations = self.parallel(
            delayed(_processSingleInput)(*args) for args in
            GeneratorProcessor._joblib_generator(self.pipelines, generator))

        for (data_source, results, ts) in evaluations:
            sink.add_frame(data_source, results, ts)
        sink.finish()

    @staticmethod
    def _joblib_generator(pipelines, generator):
        for idx, (data_source, img, ts) in enumerate(generator):
            pipeline = pipelines[idx % len(pipelines)]
            yield pipeline, data_source, img, ts


class Pipeline(object):
    def __init__(self, inputs, outputs,
                 available_stages=pipeline.stages.Stages,
                 **config):
        if len(set(inputs)) != len(inputs):
            raise Exception("Inputs are not unique: {}".format(inputs))
        if len(set(outputs)) != len(outputs):
            raise Exception("Outputs are not unique: {}".format(outputs))

        self.inputs = inputs
        self.outputs = outputs
        self.config_dict = config
        self.available_stages = available_stages
        self.required_stages = Pipeline.required_stages(self.inputs, self.outputs,
                                                        self.available_stages)
        self.stages = [self._instantiate_stage(s) for s in self.required_stages]
        self.pipeline = self._build_pipeline(self.stages)

    @staticmethod
    def required_stages(input_stages, output_stages, available_stages=pipeline.stages.Stages):
        added_stages = set()
        requirements = set(output_stages)
        while len(requirements) > 0:
            req = requirements.pop()
            if req in input_stages:
                continue

            req_fulfilled = False
            for stage in available_stages:
                if req in stage.provides:

                    for stage_req in stage.requires:
                        requirements.add(stage_req)

                    if stage not in added_stages:
                        added_stages.add(stage)

                    req_fulfilled = True

                    break

            if not req_fulfilled:
                raise RuntimeError('Unable to fulfill requirement {}'.format(req))
        return added_stages

    @staticmethod
    def config(required_stages, only_required=True):
        config = defaultdict(dict)
        for stage in required_stages:
            sig = inspect.signature(stage)
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                if param.default == Parameter.empty:
                    config[stage.__name__][name] = 'REQUIRED'
                elif not only_required:
                    config[stage.__name__][name] = param.default
        return config

    def get_config(self, only_required=True):
        return Pipeline.config(self.required_stages, only_required)

    @staticmethod
    def _get_config_parameter_line(name, param, default_pos=30):
        def get_annotation_as_str(param):
            if param.annotation != Parameter.empty:
                anno = param.annotation
                if type(anno) == str:
                    anno_str = anno
                elif hasattr(anno, '__name__'):
                    anno_str = anno.__name__
                else:
                    anno_str = str(anno)
                return " ({})".format(anno_str)
            else:
                return ""

        ss_line = "    {}{}:".format(name, get_annotation_as_str(param))

        ss_line += " " * (default_pos - len(ss_line))
        if param.default != Parameter.empty:
            ss_line += "     {}".format(param.default)

        return ss_line

    def _instantiate_stage(self, stage):
        try:
            if stage.__name__ in self.config_dict:
                return stage(**self.config_dict[stage.__name__])
            else:
                return stage()
        except TypeError:
            if stage.__name__ not in self.config_dict:
                raise KeyError(
                    "No config for stage {} set.\n".format(stage.__name__))
            stage_config = self.config_dict[stage.__name__]
            sig = inspect.signature(stage)
            missing_configs = []
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                if (param.default == Parameter.empty and
                        name not in stage_config):
                    missing_configs.append(name)

            assert missing_configs
            missing_strs = [self._get_config_parameter_line(
                name, sig.parameters[name]) for name in missing_configs]
            raise KeyError(
                "In stage {} following config is missing:\n"
                .format(stage.__name__) + "\n".join(missing_strs))

    def _build_pipeline(self, stages):
        pipeline = []
        intermediates = set(self.inputs)

        while True:
            stage_added = False
            for stage in [stage for stage in stages if stage not in pipeline]:
                if all([req in intermediates for req in stage.requires]):
                    for result in stage.provides:
                        intermediates.add(result)

                    pipeline.append(stage)
                    stage_added = True
                    break

            if not stage_added:
                if all([req in intermediates for req in self.outputs]):
                    return pipeline
                else:
                    raise RuntimeError('Unable to construct pipeline')

    def __call__(self, inputs):
        intermediates = dict(zip(self.inputs, inputs))

        for stage in self.pipeline:
            inputs = []
            for req in stage.requires:
                for intermediate, value in intermediates.items():
                    if intermediate == req:
                        inputs.append(value)
            outputs = stage(*inputs)
            if type(outputs) not in (list, tuple):
                outputs = [outputs]

            intermediates.update(dict(zip(stage.provides, outputs)))

        outputs = {}
        for intermediate, value in intermediates.items():
            if intermediate in self.outputs:
                outputs[intermediate] = value

        assert(len(outputs) == len(self.outputs))
        return outputs


def _get_cache_dir(name):
    if 'BB_PIPELINE_CACHE_DIR' in os.environ:
        cache_dir = os.environ['BB_PIPELINE_CACHE_DIR']
    else:
        cache_dir = os.path.expanduser('~/.cache/bb_pipeline')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, name)


def download_models(config):
    new_config = copy.copy(config)
    for stage_name in ('Localizer', 'Decoder', 'TagSimilarityEncoder'):
        if stage_name not in config:
            continue
        stage_config = new_config[stage_name]
        for key, value in stage_config.items():
            if key.endswith('_path') and value.startswith('http'):
                name = value.replace('/', '_')
                fname = _get_cache_dir(name)
                if not os.path.exists(fname):
                    print("Downloading {}".format(value))
                    urllib.request.urlretrieve(value, fname)
                stage_config[key] = fname
    return new_config


def get_config_from_ini(fname):
    config = configparser.ConfigParser()
    config.read(fname)
    return download_models(config)


def get_auto_config():
    """
    Returns a sane default config for the pipeline.
    """
    config_fname = os.path.join(os.path.dirname(__file__), 'config.ini')
    return get_config_from_ini(config_fname)

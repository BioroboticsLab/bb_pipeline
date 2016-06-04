
from pipeline.stages import Stages
import inspect
from inspect import Parameter

class Pipeline(object):
    def __init__(self, inputs, outputs, **config):
        self.inputs = inputs
        self.outputs = outputs
        self.config = config
        self.required_stages = self._required_stages()
        if config.get('print_config', False):
            self._print_config()
        if config.get('print_config_dict', False):
            print(self._config_dict())
        self.stages = [self._instantiate_stage(s) for s in self.required_stages]
        self.pipeline = self._build_pipeline(self.stages)

    def _required_stages(self):
        added_stages = set()
        requirements = set(self.outputs)
        while len(requirements) > 0:
            req = requirements.pop()
            if req in self.inputs:
                continue

            req_fulfilled = False
            for stage in Stages:
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

    def _print_config(self):
        required_config = []
        for stage in self.required_stages:
            ss = "{}:\n".format(stage.__name__)
            sig = inspect.signature(stage)
            has_params = False

            for name, param in sig.parameters.items():
                if name in ['self', 'config']:
                    continue
                has_params = True
                if param.default == Parameter.empty:
                    required_config.append((name, param))
                ss += self._get_config_parameter_line(name, param) + "\n"

            if has_params:
                print(ss)
        if required_config:
            print("Required:")
            for name, param in required_config:
                print(self._get_config_parameter_line(name, param))

    def _config_dict(self):
        def get_default(param):
            if param.default == Parameter.empty:
                return "'REQUIRED'"
            else:
                value = param.default
                if type(value) == str:
                    return "'{}'".format(value)
                else:
                    return str(value)

        required_config = []
        s = "{\n"
        for stage in self.required_stages:
            ss = "    # {}\n".format(stage.__name__)
            sig = inspect.signature(stage)
            has_params = False
            for name, param in sig.parameters.items():
                if name in ['self', 'config']:
                    continue
                has_params = True
                if param.default == Parameter.empty:
                    required_config.append((name, param))
                ss += "    '{}': {},\n".format(name, get_default(param))
            if has_params:
                s += ss
        s += "}"
        return s

    def _instantiate_stage(self, stage):
        try:
            return stage(**self.config)
        except TypeError as e:
            sig = inspect.signature(stage)
            missing_configs = []
            for name, param in sig.parameters.items():
                if name in ['self', 'config']:
                    continue
                if (param.default == Parameter.empty and
                        name not in self.config):
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
        intermediates = set(inputs)

        for stage in self.pipeline:
            inputs = []
            for req in stage.requires:
                # TODO: maybe use map here instead
                for intermediate in intermediates:
                    if type(intermediate) == req:
                        inputs.append(intermediate)
            outputs = stage(*inputs)
            intermediates = intermediates.union(intermediates, outputs)

        outputs = {}
        for output in self.outputs:
            for intermediate in intermediates:
                if type(intermediate) == output:
                    outputs[type(intermediate)] = intermediate

        assert(len(outputs) == len(self.outputs))
        return outputs
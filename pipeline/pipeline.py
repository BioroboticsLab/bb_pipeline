
from pipeline.objects import Filename, PipelineResult
from pipeline.stages import Stages
from bb_binary import build_frame_container
import inspect
from inspect import Parameter
from bb_binary import Cam, DataSource, FrameContainer, build_frame


class Sink:
    def add_frame(self, data_source, frame):
        raise NotImplemented()

    def finish(self):
        raise NotImplemented()


def unique_id():
    return 1


class BBBinaryRepoSink(Sink):
    def __init__(self, repo):
        self.repo = repo
        self.frames = []
        self.data_sources_fname = []
        self.data_sources = []

    def add_frame(self, data_source, results, timestamp):
        detections = results[PipelineResult]
        fname = data_source.filename
        if fname not in self.data_sources_fname:
            self.data_sources.append(data_source)
            self.data_sources_fname.append(fname)
        data_source_idx = self.data_sources_fname.index(fname)
        self.frames.append((data_source_idx, detections, timestamp))

    def finish(self):
        self.frames.sort(key=lambda x: x[2])
        start_ts = self.frames[0][2]
        end_ts = self.frames[-1][2]
        fc = FrameContainer.new_message(fromTimestamp=start_ts,
                                        toTimestamp=end_ts,
                                        id=unique_id())
        dataSources = fc.init('dataSources', len(self.data_sources))
        for i, dsource in enumerate(self.data_sources):
            dataSources[i] = dsource

        frames = fc.init('frames', len(self.frames))
        for i, (data_source_idx, detection, timestamp) in enumerate(self.frames):
            frame = frames[i]
            frame.dataSource = data_source_idx
            detections_builder = frame.detectionsUnion.init(
                'detectionsDP', len(detection.positions.positions))
            for i, db in enumerate(detections_builder):
                db.tagIdx = i
                db.xpos = int(detection.positions.positions[i, 0])
                db.ypos = int(detection.positions.positions[i, 1])
                db.xposHive = int(detection.hive_positions.positions[i, 0])
                db.yposHive = int(detection.hive_positions.positions[i, 1])
                db.zRotation = float(detection.orientations.orientations[i, 0])
                db.yRotation = float(detection.orientations.orientations[i, 1])
                db.xRotation = float(detection.orientations.orientations[i, 2])
                db.localizerSaliency = float(detection.saliencies.saliencies[i, 0])
                db.radius = float(0)
                decodedId = db.init('decodedId', len(detection.ids.ids[i]))
                for j, bit in enumerate(detection.ids.ids[i]):
                    decodedId[j] = int(round(255*bit))
        self.repo.add(fc)


class GeneratorProcessor(object):
    def __init__(self, pipeline, sink_factory):
        self.pipeline = pipeline
        self.sink_factory = sink_factory

    def __call__(self, generator):
        sink = self.sink_factory()
        for (data_source, img, ts) in generator:
            results = self.pipeline([img, ts])
            sink.add_frame(data_source, results, ts.timestamp)
        sink.finish()


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
            if type(outputs) not in (list, tuple):
                outputs = [outputs]
            intermediates = intermediates.union(intermediates, outputs)

        outputs = {}
        for output in self.outputs:
            for intermediate in intermediates:
                if type(intermediate) == output:
                    outputs[type(intermediate)] = intermediate

        assert(len(outputs) == len(self.outputs))
        return outputs
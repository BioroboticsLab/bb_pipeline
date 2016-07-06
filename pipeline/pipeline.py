import av
from datetime import datetime
from itertools import chain
import pytz
import os
from joblib import Parallel, delayed
from pipeline.objects import PipelineResult
import pipeline.stages
import inspect
from inspect import Parameter
from bb_binary import DataSource, FrameContainer, \
    parse_image_fname, parse_video_fname, get_timezone


class Sink:
    def add_frame(self, data_source, frame):
        raise NotImplemented()

    def finish(self):
        raise NotImplemented()


def unique_id():
    return 1


class BBBinaryRepoSink(Sink):
    def __init__(self, repo, camId):
        self.repo = repo
        self.frames = []
        self.data_sources_fname = []
        self.data_sources = []
        self.camId = camId

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
                                        camId=self.camId,
                                        id=unique_id())
        dataSources = fc.init('dataSources', len(self.data_sources))
        for i, dsource in enumerate(self.data_sources):
            dataSources[i] = dsource

        frames = fc.init('frames', len(self.frames))
        for i, (data_source_idx, detection, timestamp) in enumerate(self.frames):
            frame = frames[i]
            frame.dataSourceIdx = data_source_idx
            frame.frameIdx = int(i)
            frame.timestamp = timestamp
            detections_builder = frame.detectionsUnion.init(
                'detectionsDP', len(detection.positions))
            for i, db in enumerate(detections_builder):
                db.idx = i
                db.xpos = int(detection.positions[i, 0])
                db.ypos = int(detection.positions[i, 1])
                db.xposHive = int(detection.hive_positions[i, 0])
                db.yposHive = int(detection.hive_positions[i, 1])
                db.zRotation = float(detection.orientations[i, 0])
                db.yRotation = float(detection.orientations[i, 1])
                db.xRotation = float(detection.orientations[i, 2])
                db.localizerSaliency = float(detection.saliencies[i, 0])
                db.radius = float(0)
                decodedId = db.init('decodedId', len(detection.ids[i]))
                for j, bit in enumerate(detection.ids[i]):
                    decodedId[j] = int(round(255*bit))
        self.repo.add(fc)


def _processSingleInput(pipeline, data_source, img, ts):
    return data_source, pipeline([img, ts]), ts


class GeneratorProcessor(object):
    def __init__(self, pipelines, sink_factory):
        if type(pipelines) == Pipeline:
            pipelines = [pipelines]
        self.pipelines = pipelines
        self.parallel = Parallel(n_jobs=len(pipelines), backend='threading',
                                 pre_dispatch='2.*n_jobs')
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


def get_timestamps(fname_video, path_filelists, ts_format='2015'):
    def get_flist_name(ts):
        fmt = '%Y%m%d'
        dt_utc = datetime.fromtimestamp(ts, tz=pytz.utc)
        dt = dt_utc.astimezone(get_timezone())
        if ts_format == '2014':
            return dt.strftime(fmt) + '.txt'
        elif ts_format == '2015':
            return os.path.join(dt.strftime(fmt), 'images.txt')
        else:
            assert(False)

    def find_file(name, path):
        for root, dirs, files in os.walk(path):
            if name in [os.path.join(os.path.basename(root), f) for f in files]:
                return os.path.join(path, name)
        assert(False)

    cam, from_ts, to_ts = parse_video_fname(fname_video)
    txt_files = set([get_flist_name(from_ts), get_flist_name(to_ts)])
    txt_paths = [find_file(f, path_filelists) for f in txt_files]

    image_fnames = list(chain.from_iterable([open(path, 'r').readlines() for path in txt_paths]))
    first_fname = fname_video.split('_TO_')[0] + '.jpeg\n'
    second_fname = fname_video.split('_TO_')[1].split('.mkv')[0] + '.jpeg\n'
    image_fnames.sort()

    fnames = image_fnames[image_fnames.index(first_fname):image_fnames.index(second_fname) + 1]
    return [parse_image_fname(fn, format='beesbook')[1] for fn in fnames]


def video_generator(path_video, path_filelists):
    fname_video = os.path.basename(path_video)
    timestamps = get_timestamps(fname_video, path_filelists)
    data_source = DataSource.new_message(filename=fname_video)

    container = av.open(path_video)
    assert(len(container.streams) == 1)
    video = container.streams[0]

    idx = 0
    for packet in container.demux(video):
        for frame in packet.decode():
            img = frame.to_rgb().to_nd_array()[:, :, 0]
            yield data_source, img, timestamps[idx]
            idx += 1


class Pipeline(object):
    def __init__(self, inputs, outputs,
                 available_stages=pipeline.stages.Stages,
                 **config):
        self.inputs = inputs
        self.outputs = outputs
        self.config = config
        self.available_stages = available_stages
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
            for stage in self.available_stages:
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
        except TypeError:
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

import av
import hashlib
from datetime import datetime
from itertools import chain
import uuid
import pytz
import os
from bb_binary import DataSource, FrameContainer, \
    parse_image_fname, parse_video_fname, get_timezone
from pipeline.objects import PipelineResult


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


class Sink:
    def add_frame(self, data_source, frame):
        raise NotImplemented()

    def finish(self):
        raise NotImplemented()


def unique_id():
    hasher = hashlib.sha1()
    hasher.update(uuid.uuid4().bytes)
    # strip to 64 bits
    hash = int.from_bytes(hasher.digest(), byteorder='big') >> 95
    return hash


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

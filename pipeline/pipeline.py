#!/usr/bin/python3

from pipeline.stages import Stages


class Pipeline(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        stages = self._instanciate_required_stages()
        self.pipeline = self._build_pipeline(stages)

    def _instanciate_required_stages(self):
        stage_instances = set()
        added_stages = set()
        requirements = set(self.outputs)

        while len(requirements) > 0:
            req = requirements.pop()

            if req in self.inputs:
                continue

            req_fulfilled = False
            for stage in Stages:
                if req in stage.provides:
                    stage_instance = stage()

                    for stage_req in stage.requires:
                        requirements.add(stage_req)

                    if stage not in added_stages:
                        added_stages.add(stage)
                        stage_instances.add(stage_instance)

                    req_fulfilled = True

                    break

            if not req_fulfilled:
                raise RuntimeError('Unable to fulfill requirement {}'.format(req))

        return stage_instances

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
class PipelineStage(object):
    requires = []
    provides = []

    def __init__(self):
        pass

    def __call__(self, *inputs):
        assert len(self.requires) == len(inputs)
        for required, input in zip(self.requires, inputs):
            if hasattr(required, 'validate'):
                required.validate(input)

        outputs = self.call(*inputs)
        if type(outputs) not in [tuple, list]:
            assert len(self.provides) == 1, "If there are multiple outputs, "\
                "then they must be returned as list or tuple! But got {}.".format(outputs)
            outputs = (outputs, )

        assert len(self.provides) == len(outputs)
        for provided, output in zip(self.provides, outputs):
            if hasattr(provided, 'validate'):
                provided.validate(output)
        return outputs

    def call(self, *inputs):
        raise NotImplemented()

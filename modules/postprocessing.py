"""Class for BRATS 2017 data postprocessing."""


class PostprocessorBRATS(object):
    """Class for BRATS 2017 data postprocessing."""

    def postprocess(self, db):
        raise NotImplementedError()

    def name(self):
        """Class name reproduction."""
        return "%s()" % (type(self).__name__)

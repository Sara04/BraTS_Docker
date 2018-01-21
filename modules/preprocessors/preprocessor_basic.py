"""Class for BRATS 2017 data preprocessing."""


from .. import PreprocessorBRATS


class PreprocessorBRATSBasic(PreprocessorBRATS):
    """Class for BRATS 2017 data preprocessing."""

    """
        Attributes:
            norm_type: normalization type (mean_std or min_max)
            clip: flag indicating whether to clip values after normalization
            clip_u: upper clip value
            clip_l: lower clip value
    """
    def __init__(self, norm_type,
                 clip=True, clip_l=-2.0, clip_u=2.0):
        """Initialization of PreprocessorBRATSBasic attributes."""
        self.norm_type = norm_type
        self.clip = clip
        self.clip_l = clip_l
        self.clip_u = clip_u

    def name(self):
        """Class name reproduction."""
        """
            Returns:
                PreprocessingBRATS object's name
        """
        if not self.clip:
            return "%s(norm_type=%s)" % (type(self).__name__, self.norm_type)
        else:
            return ("%s(norm_type=%s, clip_l=%s, clip_u=%s)"
                    % (type(self).__name__, self.norm_type,
                       self.clip_l, self.clip_u))

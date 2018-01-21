"""Class for BRATS 2017 data postprocessing."""

import numpy as np
import os
import nibabel as nib
from scipy.ndimage import measurements
from scipy.ndimage import morphology

from .. import PostprocessorBRATS


class PostprocessorBRATSForCNN12(PostprocessorBRATS):
    """Class for BRATS 2017 data postprocessing."""

    def __init__(self, size_th_1=3000, size_th_2=1, score_th_1=0.915, score_th_2=0.4):
        """Initialization of PostprocessorBRATS attributes."""
        self.size_th_1 = size_th_1
        self.size_th_2 = size_th_2

        self.score_th_1 = score_th_1
        self.score_th_2 = score_th_2

    def postprocess(self, probabilities):

        segment_tmp = np.zeros((240, 240, 155, 5))
        segment_tmp[:, :, :, 0] = probabilities[0]
        segment_tmp[:, :, :, 1] = probabilities[1]
        segment_tmp[:, :, :, 2] = probabilities[2]
        segment_tmp[:, :, :, 3] = -1.0
        segment_tmp[:, :, :, 4] = probabilities[3]
        segment_test = np.argmax(segment_tmp, axis=3)

        segment_sc_test_t = probabilities[1] + probabilities[2] + probabilities[3]
        segment_sc_test_m = np.maximum(np.maximum(probabilities[1],
                                                  probabilities[2]),
                                       probabilities[3])

        mask_whole = (segment_test != 0) *\
            (segment_sc_test_t > self.score_th_1) *\
            (segment_sc_test_m > self.score_th_2)

        M, label = measurements.label(mask_whole)
        for i in range(1, label + 1):
            p = (M == i)
            p_sum = np.sum(p)
            if p_sum < self.size_th_1:
                mask_whole[p] = 0

        se = np.ones((3, 3, 3))
        mask_whole = morphology.binary_closing(mask_whole, se)
        segment_test *= mask_whole

        return segment_test

    def name(self):
        """Class name reproduction."""
        return "%s()" % (type(self).__name__)

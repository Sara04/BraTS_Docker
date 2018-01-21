"""Class for BRATS 2017 segmentation."""


class SegmentatorBRATS(object):
    """Class for BRATS 2017 segmentation."""

    """
        Methods:
            training_and_validation: training and validation of
                algorithm on training dataset
            compute_classification_scores: computation of
                classification scores
            save_model: saving trained model
            restore_model: restoreing trained model
    """

    def restore_model(self, input_path, it=0):
        """Restoring trained segmentation model."""
        """
            Arguments:
                input_path: path to the input directory
                it: iteration number if algorithm's training is iterative
        """
        raise NotImplementedError()

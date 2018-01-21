"""Class for BRATS 2017 patch extraction."""


class PatchExtractorBRATS(object):
    """Class for BRATS 2017 patch extraction."""

    """
        Attributes:
            w, h, d: volume's widht, height and depth (number of slices)

        Methods:
            extract_test_patches: extract data for testing
    """
    def __init__(self, w=240, h=240, d=155, augment_train=False):
        """Initialization of PatchExtractorBRATS attributes."""
        self.w, self.h, self.d = [w, h, d]

    def extract_test_patches(self, scan, db, pp, volumes, ind_part):
        """Extraction of test patches."""
        """
            Arguments:
                scan: selected scan
                db: DatabaseBRATS object
                pp: PreprocessorBRATS object
                volumes: scan volumes
                ind_part: list of voxel indices at which patches will be
                    extracted
            Returns:
                extracted test patches
        """
        raise NotImplementedError()

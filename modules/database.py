"""Class for BRATS 2017 database management."""


class DatabaseBRATS(object):
    """Class for BRATS 2017 database management."""

    """
        Attributes:
            db_path: path where training and validation subsets are stored
            n_classes: number of tumor subregions
            classes: list of tumor subregions, namely
                0 - non tumorous tissue
                1 -  necrotic and non-enhancing tumor
                2 - peritumoral edema
                4 - enhancing tumor
            n_modalities: number of MRI modalities
            modalities: list of MRI modalities, namely
                t1 - T1 weighted image, spin-lattice relaxation time
                t2 - T2 weighted image, spin-splin relaxation time
                t1ce - contrast enhanced T1 weighted image
                flair - T2 weighted fluid attenuation inversion recovery

            h, w, d: scan's height, width and depth (number of slices)

            valid_p: percentage of training data that will be used for
                algorithm's training validation

    """
    def __init__(self, db_path, n_classes=4, classes=[0, 1, 2, 4],
                 n_modalities=4, modalities=['t1', 't2', 't1ce', 'flair'],
                 h=240, w=240, d=155, valid_p=0.2):
        """Initialization of DatabaseBRATS attributes."""
        self.db_path = db_path
        self.n_classes = n_classes
        self.classes = classes
        self.n_modalities = n_modalities
        self.modalities = modalities

        self.h, self.w, self.d = [h, w, d]

        self.valid_p = valid_p

    def name(self):
        """Return database name."""
        return "%s(valid_p=%s)" % (type(self).__name__, self.valid_p)

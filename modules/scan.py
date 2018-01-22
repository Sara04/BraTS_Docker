"""Class for BRATS 2017 scan/sample info loading and generating."""
import os
import nibabel as nib
import numpy as np


class ScanBRATS(object):
    """Class for BRATS 2017 scan info loading and generating."""

    """
        Attributes:
            name: scan name
            relative_path: scan path relative to the database path
            mode: train, valid or test mode (dataset)

        Methods:
            load_volume: load a scan of chosen modality
            load_normalized_volume: load scan of a chosen modality with
                normalized values
            load_brain_mask: load brain and skul region mask
            load_tumor_distance_map: load distances to tumorous tissue
            load_volumes: load all modalities of a scan and
                segmentation, brain mask and tumor distance map optionally
    """
    def __init__(self, clip=[-2.0, 2.0],
                 scan_modalities=['t1', 't2', 't1c', 'flair']):
        """Initialization of ScanBRATS attributes."""
        self.extension = None
        self.clip = clip
        self.scan_modalities = scan_modalities

    def load_signle_scan(self, path):
        """Loading all volumes as a list numpy arrays."""

        if os.path.exists(os.path.join(path, 't1.nii.gz')):
            self.extension = '.nii.gz'
        else:
            self.extension = '.nii'

        scans = []
        for m in self.scan_modalities:
            volume_path = os.path.join(path, m + self.extension)
            scans.append(nib.load(volume_path).get_data().astype('float32'))

        bm = (scans[0] != 0) * (scans[1] != 0) * (scans[2] != 0) * (scans[3] != 0)
        scans.append(bm)

        for m_idx, m in enumerate(self.scan_modalities):
            non_zeros = scans[m_idx] != 0.0
            mean_ = np.mean(scans[m_idx][non_zeros == 1])
            std_ = np.std(scans[m_idx][non_zeros == 1])

            scans[m_idx] = np.clip(scans[4] * (scans[m_idx] - mean_) / std_,
                                   self.clip[0], self.clip[1])

        return scans

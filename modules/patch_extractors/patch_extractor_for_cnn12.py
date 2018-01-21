"""Class for BRATS 2017 patch extraction."""

import numpy as np
from .. import PatchExtractorBRATS


class PatchExtractorBRATSForCNN12(PatchExtractorBRATS):
    """Class for BRATS 2017 patch extraction for cnn segmentators."""

    """
        Attributes:
            w, h, d: volume's widht, height and depth (number of slices)
            lp_w, lp_h, lp_d: width, height and depth of large region
                patches
            sp_w, sp_h, sp_d: width, height and depth of small region
                patches
            td_th_1, td_th_2: thresholds used to determine small and large
                neighbourhoods of tumor
            lpm_d: depth of large region patches per modality
            spm_d: depth of small region patches per modality

        Methods:
            extract_test_patches: extract data for testing
    """
    def __init__(self,
                 scans_per_batch=5, patches_per_scan=50,
                 test_patches_per_scan=100,
                 lp_w=45, lp_h=45, lp_d=11, sp_w=17, sp_h=17, sp_d=4,
                 td_th_1=20, td_th_2=256, lpm_d=2, spm_d=1, **kwargs):
        """Initialization of PatchExtractorBRATS attributes."""
        super(PatchExtractorBRATSForCNN12, self).__init__(**kwargs)
        self.scans_per_batch = scans_per_batch
        self.patches_per_scan = patches_per_scan
        self.test_patches_per_scan = test_patches_per_scan

        self.lp_w, self.lp_h, self.lp_d = [lp_w, lp_h, lp_d]
        self.sp_w, self.sp_h, self.sp_d = [sp_w, sp_h, sp_d]

        self.td_th_1 = td_th_1
        self.td_th_2 = td_th_2

        self.pvs, self.pve = [(self.lp_h - 1) / 2, (self.lp_h + 1) / 2]
        self.phs, self.phe = [(self.lp_w - 1) / 2, (self.lp_w + 1) / 2]

        self._get_coordinates()

        self.lpm_d = lpm_d
        self.spm_d = spm_d

    def _get_coordinates(self):

        self.h_coord = np.zeros((self.h, self.w))
        self.v_coord = np.zeros((self.h, self.w))
        self.d_coord = np.zeros((self.h, self.w, self.d))
        wh, hh, dh = [self.w / 2, self.h / 2, self.d / 2]
        for r_idx in range(self.h):
            for c_idx in range(self.w):
                self.h_coord[r_idx, c_idx] = float(c_idx - wh) / wh
                self.v_coord[r_idx, c_idx] = float(r_idx - hh) / hh
                for d_idx in range(self.d):
                    self.d_coord[r_idx, c_idx, d_idx] = float(d_idx - dh) / dh

    def _allocate_data_memory(self, db):
        data = {'region_1': {}, 'region_2': {}}
        for i in db.classes:
            data['region_1'][i] = {}
            data['region_1'][i]['l_patch'] =\
                np.zeros((self.patches_per_scan * self.scans_per_batch,
                          self.lp_w * self.lp_h * self.lp_d))
            data['region_1'][i]['s_patch'] =\
                np.zeros((self.patches_per_scan * self.scans_per_batch,
                          self.sp_w * self.sp_h * self.sp_d))
        for i in range(2):
            data['region_2'][i] = {}
            data['region_2'][i]['l_patch'] =\
                np.zeros((self.patches_per_scan * self.scans_per_batch,
                          self.lp_w * self.lp_h * self.lp_d))
            data['region_2'][i]['s_patch'] =\
                np.zeros((self.patches_per_scan * self.scans_per_batch,
                          self.sp_w * self.sp_h * self.sp_d))
        return data

    def _extract_distances_for_point(self, b):

        dist = np.zeros((self.lp_h, self.lp_w, 3))

        b_, d_ = [np.copy(b), [0, self.lp_h, 0, self.lp_w]]
        b_, d_ = self._verify_border_cases(b_, d_)

        dist[d_[0]:d_[1], d_[2]:d_[3], 0] =\
            self.h_coord[b[0]: b[1], b[2]: b[3]]
        dist[d_[0]:d_[1], d_[2]:d_[3], 0] =\
            self.v_coord[b[0]: b[1], b[2]: b[3]]
        dist[d_[0]:d_[1], d_[2]:d_[3], 0] =\
            self.d_coord[b[0]: b[1], b[2]: b[3], b[4]]
        return dist

    def _verify_border_cases(self, b_, d_):

        if b_[0] < 0:
            d_[0], b_[0] = [0 - b_[0], 0]
        if b_[2] < 0:
            d_[2], b_[2] = [0 - b_[2], 0]
        if b_[1] > self.h:
            d_[1], b_[1] = [self.lp_h - (b_[1] - self.h), self.h]
        if b_[3] > self.w:
            d_[3], b_[3] = [self.lp_w - (b_[3] - self.w), self.w]

        return b_, d_

    def _modality_patches(self, scan, m, volume, b):

        lpm = np.zeros((self.lp_h, self.lp_w, self.lpm_d))
        spm = np.zeros((self.sp_h, self.sp_w, self.spm_d))

        b_, d_ = [np.copy(b), [0, self.lp_h, 0, self.lp_w]]
        b_, d_ = self._verify_border_cases(b_, d_)

        lpm[d_[0]:d_[1], d_[2]:d_[3], 0] =\
            volume[b_[0]: b_[1], b_[2]: b_[3], b_[4]]

        lpm[self.lp_h - d_[1]:self.lp_h - d_[0],
            self.lp_w - d_[3]:self.lp_w - d_[2], 1] =\
            volume[self.h - b_[1]: self.h - b_[0], b_[2]: b_[3], b_[4]]

        spm[:, :, 0] = lpm[:, :, 0][(self.lp_h - 1) / 2 - (self.sp_h - 1) / 2:
                                    (self.lp_h - 1) / 2 + (self.sp_h + 1) / 2,
                                    (self.lp_w - 1) / 2 - (self.sp_w - 1) / 2:
                                    (self.lp_w - 1) / 2 + (self.sp_w + 1) / 2]
        return lpm, spm

    def extract_test_patches(self, scan, db, volumes, ind_part):
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
        n_indices = len(ind_part[0])
        test_data = {}
        test_data['l_patch'] =\
            np.zeros((n_indices, self.lp_h * self.lp_w * self.lp_d))
        test_data['s_patch'] =\
            np.zeros((n_indices, self.sp_h * self.sp_w * self.sp_d))

        lp = np.zeros((self.lp_h, self.lp_w, self.lp_d))
        sp = np.zeros((self.sp_h, self.sp_w, self.sp_d))
        for j in range(n_indices):
            b = [ind_part[0][j] - self.pvs, ind_part[0][j] + self.pve,
                 ind_part[1][j] - self.phs, ind_part[1][j] + self.phe,
                 ind_part[2][j]]
            for i, m in enumerate(db.modalities):
                lpm, spm = self._modality_patches(scan, m, volumes[i], b)
                lp[:, :, i * self.lpm_d:(i + 1) * self.lpm_d] = lpm
                sp[:, :, i * self.spm_d:(i + 1) * self.spm_d] = spm
            lp[:, :, self.lpm_d * db.n_modalities:
               (self.lpm_d * db.n_modalities + 3)] =\
                self._extract_distances_for_point(b)
            test_data['l_patch'][j, :] = np.ravel(lp)
            test_data['s_patch'][j, :] = np.ravel(sp)
        return test_data

    def name(self):
        """Class name reproduction."""
        """
            Returns patch_extractor's name.
        """
        return ("%s(lp_w=%s, lp_h=%s, lp_d=%s, sp_w=%s, sp_h=%s, sp_d=%s, "
                "td_th_1=%s, td_th_2=%s)"
                % (type(self).__name__,
                   self.lp_w, self.lp_h, self.lp_d,
                   self.sp_w, self.sp_h, self.sp_d,
                   self.td_th_1, self.td_th_2))

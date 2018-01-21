"""Toolchain for algorithm's training."""
import argparse
import sys
import os
import nibabel as nib
import numpy as np
from modules.scan import ScanBRATS

from create_modules_objects import create_modules_objects_from_config

MODEL_NO = 13000


def main():

    """Function that runs testing of a BRATS algorithm."""
    # _______________________________________________________________________ #
    parser = argparse.ArgumentParser(description='An algorithm for '
                                     'Brain Tumor Segmentation Challenge')

    parser.add_argument('-image_id', dest='DOCKER_IMAGE_ID', required=True,
                        help='Docker\'s image id.',)
    parser.add_argument('-config', dest='config_path', required=False,
                        default='config_files/config.json',
                        help='Path to the configuration file.')

    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print "\nConfiguration file does not exist!\n"
        sys.exit(2)

    modules_objects = create_modules_objects_from_config(args.config_path)
    # _______________________________________________________________________ #

    # 1. Loading modules' objects
    # _______________________________________________________________________ #
    db = modules_objects['database']
    prep = modules_objects['preprocessor']
    patch_ex = modules_objects['patch_extractor']
    seg = modules_objects['segmentator']
    post = modules_objects['postprocessor']

    scan = ScanBRATS()
    scans = scan.load_signle_scan('/data')

    # _______________________________________________________________________ #

    # 2. Loading segmentator
    # _______________________________________________________________________ #
    seg_path = os.path.join('model/segmentators', db.name(), prep.name(),
                            patch_ex.name(), seg.name())
    seg.restore_model(seg_path, MODEL_NO)

    # _______________________________________________________________________ #

    # 3. Computing class probabilities
    # _______________________________________________________________________ #
    probabilities = seg.compute_clf_scores(db, scan, patch_ex, scans)

    for c_idx, c in enumerate(db.classes):

            probabilities[c_idx].astype('float32')

            prob_nii = nib.Nifti1Image(probabilities[c_idx], np.eye(4))

            output_path = os.path.join('/data/results/',
                                       'tumor_' + args.DOCKER_IMAGE_ID +
                                       '_prob_' + str(c) + scan.extension)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            prob_nii.to_filename(output_path)
    # _______________________________________________________________________ #

    # 4. Post-processing and computation of classes
    # _______________________________________________________________________ #
    segmentation = post.postprocess(probabilities)

    segmentation.astype('int16')

    segmentation = nib.Nifti1Image(segmentation, np.eye(4))

    output_path = os.path.join('/data/results/',
                               'tumor_' + args.DOCKER_IMAGE_ID +
                               '_class' + scan.extension)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    segmentation.to_filename(output_path)


if __name__ == '__main__':
    main()

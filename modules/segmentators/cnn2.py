"""Segmentation class."""
import tensorflow as tf
import os
import numpy as np
import time

from .. import SegmentatorBRATS
from cnn_utils import _weight_variable_xavier, _bias_variable
from cnn_utils import _conv, _max_pool


class CnnBRATS2(SegmentatorBRATS):
    """Segmentation class CnnBRATS2."""

    """
        Arguments:
            lr: learning rate
            lw: two and four classification problems loss weights
            kp: keep probability for drop out regularization
            restore: flag indicating whether to restore model
            restore_it: train iteration of the model to be restored
            lp_w, lp_h, lp_d: large patch width, height and depth
                (8 volume patches and 3 distance maps patches)
            sp_w, sp_h, sp_d: small patch width, height and depth
                (4 volume patches)

        Methods:
            training_and_validation: training and validation of
                algorithm on training dataset
            compute_classification_scores: computation of
                classification scores
            save_model: saving trained model
            restore_model: restoreing trained model
    """
    def __init__(self, lr=1e-4, lw=[0.25, 0.75], kp=0.5,
                 restore=True, restore_it=13000,
                 train_iters=100000,
                 lp_w=45, lp_h=45, lp_d=11, sp_w=17, sp_h=17, sp_d=4):
        """Class initialization."""
        self.lr, self.lw, self.kp = [lr, lw, kp]

        self.restore, self.restore_it = [restore, restore_it]

        self.train_iters = train_iters

        self.keep_prob = tf.placeholder(tf.float32)
        self.lp_w, self.lp_h, self.lp_d = [lp_w, lp_h, lp_d]
        self.sp_w, self.sp_h, self.sp_d = [sp_w, sp_h, sp_d]

        self.lp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.lp_h * self.lp_w * self.lp_d])
        self.sp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.sp_h * self.sp_w * self.sp_d])
        self.gt_r1 = tf.placeholder(tf.float32, shape=[None, 4])

        self.lp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.lp_h * self.lp_w * self.lp_d])
        self.sp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.sp_h * self.sp_w * self.sp_d])
        self.gt_r2 = tf.placeholder(tf.float32, shape=[None, 2])

        self.sess = tf.Session()

        lp_imgs_r1 = tf.reshape(self.lp_x_r1,
                                [-1, self.lp_h, self.lp_w, self.lp_d])
        sp_imgs_r1 = tf.reshape(self.sp_x_r1,
                                [-1, self.sp_h, self.sp_w, self.sp_d])

        lp_imgs_r2 = tf.reshape(self.lp_x_r2,
                                [-1, self.lp_h, self.lp_w, self.lp_d])
        sp_imgs_r2 = tf.reshape(self.sp_x_r2,
                                [-1, self.sp_h, self.sp_w, self.sp_d])

        with tf.variable_scope('l_patches'):
            with tf.variable_scope('layer_1'):
                lp_w_c1 = _weight_variable_xavier([5, 5, 11, 32])
                lp_b_c1 = _bias_variable([32])
                lp_h_c1_r1 = tf.nn.relu(_conv(lp_imgs_r1, lp_w_c1) + lp_b_c1)
                lp_h_c1_r2 = tf.nn.relu(_conv(lp_imgs_r2, lp_w_c1) + lp_b_c1)
                lp_h_p1_r1 = _max_pool(lp_h_c1_r1)
                lp_h_p1_r2 = _max_pool(lp_h_c1_r2)
            with tf.variable_scope('layer_2'):
                lp_w_c2 = _weight_variable_xavier([5, 5, 32, 64])
                lp_b_c2 = _bias_variable([64])
                lp_h_c2_r1 = tf.nn.relu(_conv(lp_h_p1_r1, lp_w_c2) + lp_b_c2)
                lp_h_c2_r2 = tf.nn.relu(_conv(lp_h_p1_r2, lp_w_c2) + lp_b_c2)
                lp_h_p2_r1 = _max_pool(lp_h_c2_r1)
                lp_h_p2_r2 = _max_pool(lp_h_c2_r2)
            with tf.variable_scope('layer_3'):
                lp_w_c3 = _weight_variable_xavier([5, 5, 64, 128])
                lp_b_c3 = _bias_variable([128])
                lp_h_c3_r1 = tf.nn.relu(_conv(lp_h_p2_r1, lp_w_c3) + lp_b_c3)
                lp_h_c3_r2 = tf.nn.relu(_conv(lp_h_p2_r2, lp_w_c3) + lp_b_c3)
                lp_h_c3_fl_r1 = tf.reshape(lp_h_c3_r1, [-1, 4 * 4 * 128])
                lp_h_c3_fl_r2 = tf.reshape(lp_h_c3_r2, [-1, 4 * 4 * 128])
            with tf.variable_scope('layer_4'):
                lp_w_fcn1 = _weight_variable_xavier([4 * 4 * 128, 128])
                lp_b_fcn1 = _bias_variable([128])
                lp_h_fcn1_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_c3_fl_r1,
                                                       lp_w_fcn1) +
                                             lp_b_fcn1),
                                  self.keep_prob)
                lp_h_fcn1_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_c3_fl_r2,
                                                       lp_w_fcn1) +
                                             lp_b_fcn1),
                                  self.keep_prob)
            with tf.variable_scope('layer_5'):
                lp_w_fcn2 = _weight_variable_xavier([128, 32])
                lp_b_fcn2 = _bias_variable([32])
                lp_h_fcn2_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_fcn1_r1,
                                                       lp_w_fcn2) +
                                             lp_b_fcn2),
                                  self.keep_prob)
                lp_h_fcn2_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_fcn1_r2,
                                                       lp_w_fcn2) +
                                             lp_b_fcn2),
                                  self.keep_prob)

        with tf.variable_scope('s_patches'):
            with tf.variable_scope('layer_1'):
                sp_w_c1 = _weight_variable_xavier([5, 5, 4, 16])
                sp_b_c1 = _bias_variable([16])
                sp_h_c1_r1 = tf.nn.relu(_conv(sp_imgs_r1, sp_w_c1) + sp_b_c1)
                sp_h_c1_r2 = tf.nn.relu(_conv(sp_imgs_r2, sp_w_c1) + sp_b_c1)
            with tf.variable_scope('layer_2'):
                sp_w_c2 = _weight_variable_xavier([5, 5, 16, 32])
                sp_b_c2 = _bias_variable([32])
                sp_h_c2_r1 = tf.nn.relu(_conv(sp_h_c1_r1, sp_w_c2) + sp_b_c2)
                sp_h_c2_r2 = tf.nn.relu(_conv(sp_h_c1_r2, sp_w_c2) + sp_b_c2)
            with tf.variable_scope('layer_3'):
                sp_w_c3 = _weight_variable_xavier([5, 5, 32, 64])
                sp_b_c3 = _bias_variable([64])
                sp_h_c3_r1 = tf.nn.relu(_conv(sp_h_c2_r1, sp_w_c3) + sp_b_c3)
                sp_h_c3_r2 = tf.nn.relu(_conv(sp_h_c2_r2, sp_w_c3) + sp_b_c3)
                sp_h_c3_fl_r1 = tf.reshape(sp_h_c3_r1, [-1, 5 * 5 * 64])
                sp_h_c3_fl_r2 = tf.reshape(sp_h_c3_r2, [-1, 5 * 5 * 64])
            with tf.variable_scope('layer_4'):
                sp_w_fcn1 = _weight_variable_xavier([5 * 5 * 64, 128])
                sp_b_fcn1 = _bias_variable([128])
                sp_h_fcn1_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_c3_fl_r1,
                                                       sp_w_fcn1) +
                                             sp_b_fcn1),
                                  self.keep_prob)
                sp_h_fcn1_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_c3_fl_r2,
                                                       sp_w_fcn1) +
                                             sp_b_fcn1),
                                  self.keep_prob)
            with tf.variable_scope('layer_5'):
                sp_w_fcn2 = _weight_variable_xavier([128, 32])
                sp_b_fcn2 = _bias_variable([32])
                sp_h_fcn2_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_fcn1_r1,
                                                       sp_w_fcn2) +
                                             sp_b_fcn2),
                                  self.keep_prob)
                sp_h_fcn2_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_fcn1_r2,
                                                       sp_w_fcn2) +
                                             sp_b_fcn2),
                                  self.keep_prob)

        with tf.variable_scope('patch_merge'):
            with tf.variable_scope('layer_1'):
                feat_r1 = tf.concat([lp_h_fcn2_r1, sp_h_fcn2_r1], 1)
                feat_r2 = tf.concat([lp_h_fcn2_r2, sp_h_fcn2_r2], 1)
                mp_w_fcn1 = _weight_variable_xavier([64, 32])
                mp_b_fcn1 = _bias_variable([32])
                mp_h_fcn1_r1 = tf.nn.relu(tf.matmul(feat_r1, mp_w_fcn1) +
                                          mp_b_fcn1)
                mp_h_fcn1_r2 = tf.nn.relu(tf.matmul(feat_r2, mp_w_fcn1) +
                                          mp_b_fcn1)
            with tf.variable_scope('layer_2'):
                mp_w_fcn2 = _weight_variable_xavier([32, 4])
                mp_b_fcn2 = _bias_variable([4])
                mp_h_fcn1_r1 =\
                    tf.nn.softmax(tf.matmul(mp_h_fcn1_r1, mp_w_fcn2) +
                                  mp_b_fcn2)
                mp_h_fcn1_r2_4 =\
                    tf.nn.softmax(tf.matmul(mp_h_fcn1_r2, mp_w_fcn2) +
                                  mp_b_fcn2)
                mp_4_to_2 = tf.Variable([[1., 0], [0, 1.], [0, 1.], [0, 1.]])
                mp_h_fcn1_r2 = tf.matmul(mp_h_fcn1_r2_4, mp_4_to_2)

        # ___________________________________________________________________ #
        cross_entropy =\
            self.lw[0] *\
            tf.reduce_mean(-tf.reduce_sum(self.gt_r1 * tf.log(mp_h_fcn1_r1),
                                          reduction_indices=[1])) +\
            self.lw[1] *\
            tf.reduce_mean(-tf.reduce_sum(self.gt_r2 * tf.log(mp_h_fcn1_r2),
                                          reduction_indices=[1]))
        self.train_step =\
            tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

        clf_r1 = tf.equal(tf.argmax(mp_h_fcn1_r1, 1), tf.argmax(self.gt_r1, 1))
        self.accuracy_r1 = tf.reduce_mean(tf.cast(clf_r1, tf.float32))
        self.probabilities_1 = mp_h_fcn1_r1

        clf_r2 = tf.equal(tf.argmax(mp_h_fcn1_r2, 1), tf.argmax(self.gt_r2, 1))
        self.accuracy_r2 = tf.reduce_mean(tf.cast(clf_r2, tf.float32))
        self.probabilities_2 = mp_h_fcn1_r2

        vars_to_save = tf.trainable_variables()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(vars_to_save, max_to_keep=100000)

    def compute_clf_scores(self, db, scan, patch_ex, scans):

        time_start = time.time()
        indices = np.where(scans[4])
        p_0 = np.zeros((db.h, db.w, db.d))
        p_1 = np.zeros((db.h, db.w, db.d))
        p_2 = np.zeros((db.h, db.w, db.d))
        p_4 = np.zeros((db.h, db.w, db.d))

        n_indices = len(indices[0])
        i = 0
        while i < n_indices:
            s_idx = [indices[0][i:i + patch_ex.test_patches_per_scan],
                     indices[1][i:i + patch_ex.test_patches_per_scan],
                     indices[2][i:i + patch_ex.test_patches_per_scan]]
            i += patch_ex.test_patches_per_scan
            patches = patch_ex.extract_test_patches(scan, db,
                                                    scans, s_idx)
            labels =\
                self.sess.run(self.probabilities_1,
                              feed_dict={self.lp_x_r1: patches['l_patch'],
                                         self.sp_x_r1: patches['s_patch'],
                                         self.keep_prob: 1.0})
            for j in range(labels.shape[0]):
                p_0[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 0]
                p_1[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 1]
                p_2[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 2]
                p_4[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 3]

        time_end = time.time()
        print "time elapsed:", time_end - time_start
        return [p_0, p_1, p_2, p_4]

    def restore_model(self, input_path, it):
        """Restoring trained segmentation model."""
        """
            Arguments:
                input_path: path to the input directory
                it: train iteration of the model to be restored
        """
        model_path = os.path.join(input_path, 'model_' + str(it))
        self.saver.restore(self.sess, model_path)

    def name(self):
        """Class name reproduction."""
        """
            Returns segmentator's name.
        """
        return ("%s(lp_w=%s, lp_h=%s, lp_d=%s, sp_w=%s, sp_h=%s, sp_d=%s)"
                % (type(self).__name__,
                   self.lp_w, self.lp_h, self.lp_d,
                   self.sp_w, self.sp_h, self.sp_d))

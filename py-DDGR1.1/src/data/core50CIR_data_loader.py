
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pickle as pkl
import os
import logging
from hashlib import md5
from PIL import Image


class CORE50(object):

    nbatch = {
        'ni': 8,
        'nc': 9,
        'nic': 79,
        'nicv2_79': 79,
        'nicv2_196': 196,
        'nicv2_391': 391
    }

    def __init__(self, root='', preload=False, scenario='ni', cumul=False,
                 run=0, start_batch=0):

        self.root = root
        self.preload = preload
        self.scenario = scenario
        self.cumul = cumul
        self.run = run
        self.batch = start_batch

        if preload:
            print("Loading data...")
            bin_path = os.path.join(self.root, 'core50_imgs.bin')
            if os.path.exists(bin_path):
                with open(bin_path, 'rb') as f:
                    self.x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(164866, 128, 128, 3)

            else:
                with open(os.path.join(self.root, 'core50_imgs.npz'), 'rb') as f:
                    npzfile = np.load(f)
                    self.x = npzfile['x']
                    print("Writing bin for fast reloading...")
                    self.x.tofile(bin_path)

        print("Loading paths...")
        with open(os.path.join(self.root, 'paths.pkl'), 'rb') as f:
            self.paths = pkl.load(f)

        print("Loading LUP...")
        with open(os.path.join(self.root, 'LUP.pkl'), 'rb') as f:
            self.LUP = pkl.load(f)

        print("Loading labels...")
        with open(os.path.join(self.root, 'labels.pkl'), 'rb') as f:
            self.labels = pkl.load(f)

    def __iter__(self):
        return self

    def __next__(self):


        scen = self.scenario
        run = self.run
        batch = self.batch

        if self.batch == self.nbatch[scen]:
            raise StopIteration


        if self.cumul:
            train_idx_list = []
            for i in range(self.batch + 1):
                train_idx_list += self.LUP[scen][run][i]
        else:
            train_idx_list = self.LUP[scen][run][batch]


        if self.preload:
            train_x = np.take(self.x, train_idx_list, axis=0)\
                      .astype(np.float32)
        else:
            print("Loading data...")

            train_paths = []
            for idx in train_idx_list:
                train_paths.append(os.path.join(self.root, self.paths[idx]))

            train_x = self.get_batch_from_paths(train_paths).astype(np.float32)


        if self.cumul:
            train_y = []
            for i in range(self.batch + 1):
                train_y += self.labels[scen][run][i]
        else:
            train_y = self.labels[scen][run][batch]

        train_y = np.asarray(train_y, dtype=np.float32)


        self.batch += 1

        return (train_x, train_y)

    def get_test_set(self):

        scen = self.scenario
        run = self.run
        test_idx_list = self.LUP[scen][run][-1]

        if self.preload:
            test_x = np.take(self.x, test_idx_list, axis=0).astype(np.float32)
        else:

            test_paths = []
            for idx in test_idx_list:
                test_paths.append(os.path.join(self.root, self.paths[idx]))

            test_x = self.get_batch_from_paths(test_paths).astype(np.float32)

        test_y = self.labels[scen][run][-1]
        test_y = np.asarray(test_y, dtype=np.float32)

        return test_x, test_y

    next = __next__

    @staticmethod
    def get_batch_from_paths(paths, compress=False, snap_dir='',
                             on_the_fly=True, verbose=False):

        log = logging.getLogger('mylogger')

        num_imgs = len(paths)
        hexdigest = md5(''.join(paths).encode('utf-8')).hexdigest()
        log.debug("Paths Hex: " + str(hexdigest))
        loaded = False
        x = None
        file_path = None

        if compress:
            file_path = snap_dir + hexdigest + ".npz"
            if os.path.exists(file_path) and not on_the_fly:
                loaded = True
                with open(file_path, 'rb') as f:
                    npzfile = np.load(f)
                    x, y = npzfile['x']
        else:
            x_file_path = snap_dir + hexdigest + "_x.bin"
            if os.path.exists(x_file_path) and not on_the_fly:
                loaded = True
                with open(x_file_path, 'rb') as f:
                    x = np.fromfile(f, dtype=np.uint8) \
                        .reshape(num_imgs, 128, 128, 3)

        if not loaded:

            x = np.zeros((num_imgs, 128, 128, 3), dtype=np.uint8)

            for i, path in enumerate(paths):
                if verbose:
                    print("\r" + path + " processed: " + str(i + 1), end='')
                x[i] = np.array(Image.open(path))

            if verbose:
                print()

            if not on_the_fly:

                if compress:
                    with open(file_path, 'wb') as g:
                        np.savez_compressed(g, x=x)
                else:
                    x.tofile(snap_dir + hexdigest + "_x.bin")

        assert (x is not None), 'Problems loading data. x is None!'

        return x


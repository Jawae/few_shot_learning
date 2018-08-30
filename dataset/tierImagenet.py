"""Dataset API for tieredImageNet
Author: Eleni Triantafillou (eleni@cs.toronto.edu)

Refactored by Hongyang; barely used.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import io
import os
import csv
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import six
import pickle as pkl

import torch
from tools.utils import print_log
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from tools.utils import decompress


class TieredImageNetDataset(Dataset):
    """A few-shot learning dataset with refinement (unlabeled) training images"""
    def __init__(self, folder, split,
                 nway=5, nshot=1, num_unlabel=5, num_distractor=5, num_test=5,
                 label_ratio=0.2, shuffle_episode=False,
                 seed=0, aug_90=False, resize=84,
                 log_file=None, method='gnn'):
        """Creates a meta dataset.
        Args:
          folder:           the raw data folder
          split:            train/val/test/etc
          nway:             Int. N way classification problem, default 5.
          nshot:            Int. N-shot classification problem, default 1.
          num_unlabel:      Int. Number of unlabeled examples per class, default 2.
          num_distractor:   Int. Number of distractor classes, default 0.
          num_test:         Int. Number of query images, default 10.
          aug_90:           Bool. Whether to augment the training data by rotating 90 degrees.
          seed:             Int. Random seed.
          use_specific_labels:
                            bool. Whether to use specific or general labels.
        """

        self._splits_folder = 'dataset/tier_split'      # labels/synset info/etc (already in the repo)
        self._data_folder = folder                      # raw images
        print_log('\nsplit set: {}'.format(split))
        print_log("\tnum unlabel {}".format(num_unlabel), log_file)
        print_log("\tnum test {}".format(num_test), log_file)
        print_log("\tnum distractor {}".format(num_distractor), log_file)

        self.resize = resize
        self.log_file = log_file

        self._rnd = np.random.RandomState(seed)
        self._seed = seed
        self._split = split
        self._nway = nway
        self._num_distractor = num_distractor
        self._num_test = num_test
        self._num_unlabel = num_unlabel
        self._shuffle_episode = shuffle_episode
        self._label_ratio = label_ratio

        # self.transform = T.Compose([
        #     lambda x: Image.open(x).convert('RGB'),
        #     T.Resize((self.resize, self.resize)),
        #     T.ToTensor(),
        #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # ])

        # generate the following variables: see line 216
        # self._label_specific
        # self._label_general
        # self._label_specific_str
        # self._label_general_str
        # self._label_str
        # self._labels
        # self._images
        # self._label_split_idx
        self.read_cache()

        # Build a set for quick query.
        self._label_split_idx = np.array(self._label_split_idx)
        self._label_split_idx_set = set(list(self._label_split_idx))
        self._unlabel_split_idx = list(
            filter(lambda _idx: _idx not in self._label_split_idx_set, range(self._labels.shape[0])))
        self._unlabel_split_idx = np.array(self._unlabel_split_idx)
        if len(self._unlabel_split_idx) > 0:
            self._unlabel_split_idx_set = set(self._unlabel_split_idx)
        else:
            self._unlabel_split_idx_set = set()

        num_label_cls = len(self._label_str)
        self._num_classes = num_label_cls
        num_ex = self._labels.shape[0]
        ex_ids = np.arange(num_ex)
        self._label_idict = {}
        cnt = 0
        for cc in range(num_label_cls):
            self._label_idict[cc] = ex_ids[self._labels == cc]
            cnt += len(self._label_idict[cc])       # TODO weired: cnt should be labelled samples? or the total

        self._nshot = nshot

        # # Dictionary mapping categories to their synsets
        # self._catcode_to_syncode = self.build_catcode_to_syncode()
        # self._catcode_to_str = self.build_catcode_to_str()
        # self._syncode_to_str = self.build_syncode_to_str()

        # # Inverse dictionaries.
        num_ex = self._label_specific.shape[0]
        ex_ids = np.arange(num_ex)
        num_label_cls_specific = len(self._label_specific_str)
        self._label_specific_idict = {}
        cnt = 0
        for cc in range(num_label_cls_specific):
            self._label_specific_idict[cc] = ex_ids[self._label_specific == cc]
            cnt += len(self._label_specific_idict)   # TODO weired

    # def get_splits_folder(self):
    #     curdir = os.path.dirname(os.path.realpath(__file__))
    #     split_dir = os.path.join(curdir, "tiered_imagenet_split")
    #     if not os.path.exists(split_dir):
    #       raise ValueError("split_dir {} does not exist.".format(split_dir))
    #     return split_dir

    # def read_dataset(self):
    #     if not self.read_cache():
    #         specific_classes, general_classes = self.read_splits()
    #         label_idx_specific = []
    #         label_idx_general = []
    #         label_str_specific = []
    #         label_str_general = []
    #         data = []
    #         synset_dirs = os.listdir(self._imagenet_train_folder)
    #         for synset in tqdm(synset_dirs, desc="Reading dataset..."):
    #             if not synset in specific_classes:
    #                 continue
    #         for cat, synset_list in self._catcode_to_syncode.iteritems():
    #             if synset in synset_list:
    #                 break
    #         synset_dir_path = os.path.join(self._imagenet_train_folder, synset)
    #         img_list = os.listdir(synset_dir_path)
    #         for img_fname in img_list:
    #             fpath = os.path.join(synset_dir_path, img_fname)
    #             # TODO: change to self.transform
    #             if FLAGS.load_images:
    #                 img = cv2.imread(fpath)
    #                 img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_CUBIC)
    #                 img = np.expand_dims(img, 0)
    #                 data.append(img)
    #             else:
    #                 data_paths.append(fpath)   # TODO where is this data_paths
    #             label_idx_specific.append(len(label_str_specific))
    #             label_idx_general.append(len(label_str_general))
    #
    #         synset_name = self._syncode_to_str[synset]
    #         category_name = self._catcode_to_str[cat]
    #         label_str_specific.append(synset_name)
    #         if category_name not in label_str_general:
    #             label_str_general.append(category_name)
    #         print("Number of synsets {}".format(len(label_str_specific)))
    #         print("label_str_general {}".format(label_str_general))
    #         print("len label_str_general {}".format(len(label_str_general)))
    #         labels_specific = np.array(label_idx_specific, dtype=np.int32)
    #         labels_general = np.array(label_idx_general, dtype=np.int32)
    #         images = np.concatenate(data, axis=0)
    #         self._images = images
    #         self._label_specific = labels_specific
    #         self._label_general = labels_general
    #         self._label_specific_str = label_str_specific
    #         self._label_general_str = label_str_general
    #         self.read_label_split()
    #         self.save_cache()

    # CACHE
    def read_cache(self):
        """Reads dataset from cached pkl file."""
        cache_path_labels, cache_path_images = self.get_cache_path()

        # Decompress images.
        if not os.path.exists(cache_path_images):
            png_pkl = cache_path_images[:-4] + '_png.pkl'
            if os.path.exists(png_pkl):
                decompress(cache_path_images, png_pkl)
            else:
                return False
        if os.path.exists(cache_path_labels) and os.path.exists(cache_path_images):
            print_log("\tRead cached labels from {}".format(cache_path_labels), self.log_file)
            try:
                with open(cache_path_labels, "rb") as f:
                    data = pkl.load(f, encoding='bytes')
                    self._label_specific = data[b"label_specific"]
                    self._label_general = data[b"label_general"]
                    self._label_specific_str = data[b"label_specific_str"]
                    self._label_general_str = data[b"label_general_str"]
            except:
                with open(cache_path_labels, "rb") as f:
                    data = pkl.load(f)
                    self._label_specific = data["label_specific"]
                    self._label_general = data["label_general"]
                    self._label_specific_str = data["label_specific_str"]
                    self._label_general_str = data["label_general_str"]
            self._label_str = self._label_specific_str
            self._labels = self._label_specific

            print_log("\tRead cached images from {}".format(cache_path_images), self.log_file)
            with np.load(cache_path_images, mmap_mode="r", encoding='latin1') as data:
                self._images = data["images"]
            print_log("\tself._images.shape {}".format(self._images.shape), self.log_file)
            self.read_label_split()         # to obtain: self._label_split_idx
            return True
        else:
            return False

    def get_cache_path(self):
        """Gets cache file name."""
        cache_path = os.path.join(self._data_folder, self._split)
        cache_path_labels = cache_path + "_labels.pkl"
        cache_path_images = cache_path + "_images.npz"
        return cache_path_labels, cache_path_images

    # def save_cache(self):
    #     """Saves pkl cache."""
    #     cache_path_labels, cache_path_images = self.get_cache_path(self._seed)
    #     data = {
    #         "label_specific": self._label_specific,
    #         "label_general": self._label_general,
    #         "label_specific_str": self._label_specific_str,
    #         "label_general_str": self._label_general_str,
    #     }
    #     if not os.path.exists(cache_path_labels):
    #         with open(cache_path_labels, "wb") as f:
    #             pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
    #         print("Saved cache in location {}".format(self.get_cache_path()[0]))
    #     # Save the images
    #     if not os.path.exists(cache_path_images):
    #         np.savez(self.get_cache_path()[1], images=self._images)
    #         print("Saved the images in location {}".format(self.get_cache_path()[1]))

    # LABEL SPLIT
    # def read_splits(self):
    #     """
    #     Returns a list of labels belonging to the given split (as specified by self._split).
    #     Each element of this list is a (specific_label, general_label) tuple.
    #     """
    #     specific_label, general_label = [], []
    #     csv_path = os.path.join(self._splits_folder, self._split + '.csv')
    #     with open(csv_path, 'r') as csvfile:
    #         csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #         for i, row in enumerate(csvreader):
    #             # Sometimes there's an empty row at the bottom
    #             if len(row) == 0:
    #                 break
    #             specific_label.append(row[0])
    #             general_label.append(row[1])
    #
    #     num_specific_classes = list(set(specific_label))
    #     num_general_classes = list(set(general_label))
    #     print('Found {} synsets belonging to a total of {} categories for split {}.'.format(
    #         len(num_specific_classes), len(num_general_classes), self._split))
    #     return specific_label, general_label

    def label_split(self):
        """Gets label/unlabel image splits.
        Returns: labeled_split: List of int.
        """
        print_log('Label split using seed {:d}'.format(self._seed), self.log_file)
        rnd = np.random.RandomState(self._seed)
        num_label_cls = len(self._label_str)
        num_ex = self._labels.shape[0]
        ex_ids = np.arange(num_ex)

        labeled_split = []
        for cc in range(num_label_cls):
            cids = ex_ids[self._labels == cc]
            rnd.shuffle(cids)
            labeled_split.extend(cids[:int(len(cids) * self._label_ratio)])
        print_log("\tTotal number of classes {}".format(num_label_cls), self.log_file)
        print_log("\tLabeled split {}".format(len(labeled_split)), self.log_file)
        print_log("\tTotal image {}".format(num_ex), self.log_file)
        return sorted(labeled_split)

    def get_label_split_path(self):
        label_ratio_str = '_' + str(int(self._label_ratio * 100))
        seed_id_str = '_' + str(self._seed)
        if self._split in ['train', 'trainval']:
            cache_path = os.path.join(
                self._data_folder,
                self._split + '_labelsplit' + label_ratio_str + seed_id_str + '.txt')
        elif self._split in ['val', 'test']:
            cache_path = os.path.join(self._data_folder, self._split + '_labelsplit' + '.txt')
        return cache_path

    def read_label_split(self):
        cache_path_labelsplit = self.get_label_split_path()
        if os.path.exists(cache_path_labelsplit):
            self._label_split_idx = np.loadtxt(cache_path_labelsplit, dtype=np.int64)
        else:
            if self._split in ['train', 'trainval']:
                print_log('\tUse {}% image for labeled split.'.format(int(self._label_ratio * 100)), self.log_file)
                self._label_split_idx = self.label_split()
            elif self._split in ['val', 'test']:
                print_log('\tUse all image in labeled split, since we are in val/test', self.log_file)
                self._label_split_idx = np.arange(self._images.shape[0])
            else:
                raise ValueError('Unknown split {}'.format(self._split))
            self._label_split_idx = np.array(self.label_split(), dtype=np.int64)
            self.save_label_split()

    def save_label_split(self):
        np.savetxt(self.get_label_split_path(), self._label_split_idx, fmt='%d')

    # SYNSET NAME
    def build_catcode_to_syncode(self):
        catcode_to_syncode = {}
        csv_path = os.path.join(self._splits_folder, self._split + '.csv')
        print_log('\tcsv path is {}'.format(csv_path), self.log_file)
        with open(csv_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(csvreader):
                # Sometimes there's an empty row at the bottom
                if len(row) == 0:
                    break
                if not row[1] in catcode_to_syncode:
                    # Adding synset label row[0] to the list synsets belonging to category row[1]
                    catcode_to_syncode[row[1]] = []
                if not row[0] in catcode_to_syncode[row[1]]:
                    catcode_to_syncode[row[1]].append(row[0])
            print_log("\tCreated mapping from category to their synset codes with {} entries (meta-class).".
                      format(len(catcode_to_syncode), self.log_file))
        return catcode_to_syncode

    def build_syncode_to_str(self):
        """
        Build a mapping from synsets to the (string)
        description of the corresponding class.
        length: 1000
        """
        path_str = os.path.join(self._data_folder, "class_names.txt")
        path_synsets = os.path.join(self._data_folder, "synsets.txt")
        with open(path_str, "r") as f:
            lines_str = f.readlines()
        with open(path_synsets, "r") as f:
            lines_synsets = f.readlines()
        syn_to_str = {}
        for l_str, l_syn in zip(lines_str, lines_synsets):
            syn_to_str[l_syn.strip()] = l_str.strip()
        return syn_to_str

    def build_catcode_to_str(self):
        """length over 8k"""
        synset2words = {}
        path = os.path.join(self._splits_folder, "words.txt")
        for _, row in pd.read_fwf(path, header=None, names=['synset', 'words'], usecols=[0,1]).iterrows():
            synset2words[row.synset] = row.words
        return synset2words

    def __len__(self):
        return self._images.shape[0]

    def __getitem__(self, index):
        """Gets a new episode.
        within_category: bool. Whether or not to choose the N classes
        to all belong to the same more general category.
        (Only applicable for datasets with self._category_labels defined).

        within_category: bool. Whether or not to restrict the episode's classes
        to belong to the same general category (only applicable for JakeImageNet).
        If True, a random general category will be chosen, unless catcode is set.

        catcode: str. (e.g. 'n02795169') if catcode is provided (is not None),
        then the classes chosen for this episode will be restricted
        to be synsets belonging to the more general category with code catcode.
        """

        # if within_category or not catcode is None:
        #     assert hasattr(self, "_category_labels")
        #     assert hasattr(self, "_category_label_str")
        #     if catcode is None:
        #         # Choose a category for this episode's classes
        #         cat_idx = np.random.randint(len(self._category_label_str))
        #         catcode = self._catcode_to_syncode.keys()[cat_idx]
        #     cat_synsets = self._catcode_to_syncode[catcode]
        #     cat_synsets_str = [self._syncode_to_str[code] for code in cat_synsets]
        #     allowable_inds = []
        #     for str in cat_synsets_str:
        #         allowable_inds.append(np.where(np.array(self._label_str) == str)[0])
        #     class_seq = np.array(allowable_inds).reshape((-1))
        # else:
        num_label_cls = len(self._label_str)
        class_seq = np.arange(num_label_cls)

        self._rnd.shuffle(class_seq)

        train_img_ids = []
        train_labels = []
        test_img_ids = []
        test_labels = []

        train_unlabel_img_ids = []
        non_distractor = []

        train_labels_str = []
        test_labels_str = []

        is_training = self._split in ["train", "trainval"]
        assert is_training or self._split in ["val", "test"]

        for ii in range(self._nway + self._num_distractor):

            cc = class_seq[ii]
            # print(cc, ii < self._nway)
            _ids = self._label_idict[cc]

            # Split the image IDs into labeled and unlabeled.
            _label_ids = list(
                filter(lambda _id: _id in self._label_split_idx_set, _ids))
            _unlabel_ids = list(
                filter(lambda _id: _id not in self._label_split_idx_set, _ids))
            self._rnd.shuffle(_label_ids)
            self._rnd.shuffle(_unlabel_ids)

            # Add support set and query set (not for distractors).
            if ii < self._nway:
                train_img_ids.extend(_label_ids[:self._nshot])

                # Use the rest of the labeled image as queries, if num_test = -1.
                QUERY_SIZE_LARGE_ERR_MSG = (
                        "Query + reference should be less than labeled examples." +
                        "Num labeled {} Num test {} Num shot {}".format(
                            len(_label_ids), self._num_test, self._nshot))
                assert self._nshot + self._num_test <= len(
                    _label_ids), QUERY_SIZE_LARGE_ERR_MSG

                if self._num_test == -1:
                    if is_training:
                        num_test = len(_label_ids) - self._nshot
                    else:
                        num_test = len(_label_ids) - self._nshot - self._num_unlabel
                else:
                    num_test = self._num_test
                    if is_training:
                        assert num_test <= len(_label_ids) - self._nshot
                    else:
                        assert num_test <= len(_label_ids) - self._num_unlabel - self._nshot

                test_img_ids.extend(_label_ids[self._nshot:self._nshot + num_test])
                train_labels.extend([ii] * self._nshot)
                train_labels_str.extend([self._label_str[cc]] * self._nshot)
                test_labels.extend([ii] * num_test)
                test_labels_str.extend([self._label_str[cc]] * num_test)
                non_distractor.extend([1] * self._num_unlabel)
            else:
                non_distractor.extend([0] * self._num_unlabel)

            # Add unlabeled images here.
            if is_training:
                # Use labeled, unlabeled split here for refinement.
                train_unlabel_img_ids.extend(_unlabel_ids[:self._num_unlabel])

            else:
                # Copy test set for refinement.
                # This will only work if the test procedure is rolled out in a sequence.
                train_unlabel_img_ids.extend(
                    _label_ids[self._nshot + num_test:self._nshot + num_test + self._num_unlabel]
                )

        train_img = self.get_images(train_img_ids) / 255.0
        train_labels = np.array(train_labels)

        test_img = self.get_images(test_img_ids) / 255.0
        test_labels = np.array(test_labels)

        train_unlabel_img = self.get_images(train_unlabel_img_ids) / 255.0
        non_distractor = np.array(non_distractor)

        train_labels_str = np.array(train_labels_str)
        test_labels_str = np.array(test_labels_str)

        test_ids_set = set(test_img_ids)
        for _id in train_unlabel_img_ids:
            assert _id not in test_ids_set

        if self._shuffle_episode:
            # log.fatal('')
            # Shuffle the sequence order in an episode.
            # Very important for RNN based meta learners.
            train_idx = np.arange(train_img.shape[0])
            self._rnd.shuffle(train_idx)
            train_img = train_img[train_idx]
            train_labels = train_labels[train_idx]

            train_unlabel_idx = np.arange(train_unlabel_img.shape[0])
            self._rnd.shuffle(train_unlabel_idx)
            train_unlabel_img = train_unlabel_img[train_unlabel_idx]

            test_idx = np.arange(test_img.shape[0])
            self._rnd.shuffle(test_idx)
            test_img = test_img[test_idx]
            test_labels = test_labels[test_idx]

        # ndarrays below; haven't transferred to torch.tensor
        return [
                   train_img,
                   train_labels,
                   test_img,
                   test_labels
               ], {
            'x_unlabel': train_unlabel_img,
            'y_unlabel': non_distractor,
            'y_train_str': train_labels_str,
            'y_test_str': test_labels_str
        }    # in tensorflow, it's wrappd in Episode

    def get_images(self, inds=None):
        imgs = self._images[inds]
        return imgs
from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import numpy as np
from PIL import Image as pil_image
import pickle
from tools.io_utils import pickle_dump, pickle_load


def _process_im(path):

    img = pil_image.open(path)
    img = img.convert('RGB')
    img = img.resize((84, 84), pil_image.ANTIALIAS)
    img = np.array(img, dtype='float32')

    img = np.transpose(img, (2, 0, 1))
    img[0, :, :] -= 120.45
    img[1, :, :] -= 115.74
    img[2, :, :] -= 104.65
    img /= 127.5
    return img


class MiniImagenet(data.Dataset):
    def __init__(self, root, dataset='mini_imagenet'):

        self.root = os.path.join(root, 'miniImagenet')
        self.dataset = dataset

        if not self._check_exists_():
            self._init_folders_()
            # if self._check_decompress():
            #     self._decompress_()
            self._preprocess_()

    def _init_folders_(self):
        decompress = False
        if not os.path.exists(self.root):   # self.root = 'dataset/miniImagenet'
            os.makedirs(self.root)

        if not os.path.exists(os.path.join(self.root, 'compacted_datasets')):
            os.makedirs(os.path.join(self.root, 'compacted_datasets'))
            # decompress = True

        return decompress

    # def _check_decompress(self):
    #     return os.listdir('%s/mini_imagenet' % self.root) == []
    #
    # def _decompress_(self):
    #     print("\nDecompressing Images...")
    #     compressed_file = '%s/compressed/mini_imagenet/images.zip' % self.root
    #     if os.path.isfile(compressed_file):
    #         os.system('unzip %s -d %s/mini_imagenet/' % (compressed_file, self.root))
    #     else:
    #         raise Exception('Missing %s' % compressed_file)
    #     print("Decompressed")

    def _check_exists_(self):
        if not os.path.exists(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_train.pickle')) or not \
                os.path.exists(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_test.pickle')):
            return False
        else:
            return True

    def _get_image_paths_from_csv(self, file):    # the .csv file
        images_path, class_names = [], []
        with open(file, 'r') as f:
            f.readline()
            for line in f:
                name, class_ = line.split(',')
                class_ = class_[0:(len(class_)-1)]
                path = self.root + '/images/'+name
                images_path.append(path)
                class_names.append(class_)
        return class_names, images_path

    def _preprocess_(self):
        print('\nPreprocessing Mini-Imagenet images...')
        (class_names_train, images_path_train) = self._get_image_paths_from_csv('%s/train.csv' % self.root)
        (class_names_test, images_path_test) = self._get_image_paths_from_csv('%s/test.csv' % self.root)
        (class_names_val, images_path_val) = self._get_image_paths_from_csv('%s/val.csv' % self.root)

        keys_train = list(set(class_names_train))
        keys_test = list(set(class_names_test))
        keys_val = list(set(class_names_val))
        label_encoder = {}
        label_decoder = {}    # hyli: what's this used for?
        for i in range(len(keys_train)):
            label_encoder[keys_train[i]] = i
            label_decoder[i] = keys_train[i]
        for i in range(len(keys_train), len(keys_train)+len(keys_test)):
            label_encoder[keys_test[i-len(keys_train)]] = i
            label_decoder[i] = keys_test[i-len(keys_train)]
        for i in range(len(keys_train)+len(keys_test), len(keys_train)+len(keys_test)+len(keys_val)):
            label_encoder[keys_val[i-len(keys_train) - len(keys_test)]] = i
            label_decoder[i] = keys_val[i-len(keys_train)-len(keys_test)]

        counter = 0
        train_set = {}
        print('\ntrain set ...')
        for class_, path in zip(class_names_train, images_path_train):

            img = _process_im(path)
            if label_encoder[class_] not in train_set:
                train_set[label_encoder[class_]] = []
            train_set[label_encoder[class_]].append(img)
            counter += 1
            if counter % 1000 == 0:
                print("Counter "+str(counter) + " from " + str(len(images_path_train) + len(class_names_test) +
                                                               len(class_names_val)))
        pickle_dump(train_set, os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_train.pickle'))

        print('\ntest set ...')
        test_set = {}
        for class_, path in zip(class_names_test, images_path_test):
            img = _process_im(path)
            if label_encoder[class_] not in test_set:
                test_set[label_encoder[class_]] = []
            test_set[label_encoder[class_]].append(img)
            counter += 1
            if counter % 1000 == 0:
                print("Counter " + str(counter) + " from "+str(len(images_path_train) + len(class_names_test) +
                                                               len(class_names_val)))
        with open(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_test.pickle'), 'wb') as handle:
            pickle.dump(test_set, handle, protocol=2)

        print('\nval set ...')
        val_set = {}
        for class_, path in zip(class_names_val, images_path_val):
            img = _process_im(path)
            if label_encoder[class_] not in val_set:
                val_set[label_encoder[class_]] = []
            val_set[label_encoder[class_]].append(img)
            counter += 1
            if counter % 1000 == 0:
                print("Counter "+str(counter) + " from " + str(len(images_path_train) + len(class_names_test) +
                                                               len(class_names_val)))
        with open(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_val.pickle'), 'wb') as handle:
            pickle.dump(val_set, handle, protocol=2)

        # label_encoder = {}
        # keys = list(train_set.keys()) + list(test_set.keys())
        # for id_key, key in enumerate(keys):
        #     label_encoder[key] = id_key
        # with open(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_label_encoder.pickle'),
        #           'wb') as handle:
        #     pickle.dump(label_encoder, handle, protocol=2)

        print('Images preprocessed!')

    def load_dataset(self, partition, size=(84, 84)):
        print('Loading dataset ({})'.format(partition))
        if partition == 'train_val':
            # with open(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_%s.pickle' % 'train'),
            #           'rb') as handle:
            #     data = pickle.load(handle)
            data = pickle_load(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_%s.pickle' % 'train'))

            with open(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_%s.pickle' % 'val'),
                      'rb') as handle:
                data_val = pickle.load(handle)
            data.update(data_val)
            del data_val
        else:

            if partition == 'train':
                data = pickle_load(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_train.pickle'))
            else:
                with open(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_%s.pickle' % partition),
                          'rb') as handle:
                    data = pickle.load(handle)

        # skip = True
        # if skip:
        #     with open(os.path.join(self.root, 'compacted_datasets', 'mini_imagenet_label_encoder.pickle'),
        #               'rb') as handle:
        #         label_encoder = pickle.load(handle)
        # else:
        label_encoder = None

        # Resize images and normalize
        # fixme
        # for class_ in data:
        #     for i in range(len(data[class_])):
        #         image2resize = pil_image.fromarray(np.uint8(data[class_][i]))
        #         image_resized = image2resize.resize((size[1], size[0]))
        #         image_resized = np.array(image_resized, dtype='float32')
        #
        #         # Normalize
        #         image_resized = np.transpose(image_resized, (2, 0, 1))
        #         image_resized[0, :, :] -= 120.45  # R
        #         image_resized[1, :, :] -= 115.74  # G
        #         image_resized[2, :, :] -= 104.65  # B
        #         image_resized /= 127.5
        #
        #         data[class_][i] = image_resized

        print("Num classes " + str(len(data)))
        num_images = 0
        for class_ in data:
            num_images += len(data[class_])
        print("Num images " + str(num_images))
        return data, label_encoder

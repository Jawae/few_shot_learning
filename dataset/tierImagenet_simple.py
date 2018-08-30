import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import numpy as np
from PIL import Image
from tools.utils import print_log, decompress
import pickle as pkl


class tierImagenet(Dataset):
    """directly refactored from the miniImagenet.py file"""
    def __init__(self, root, mode, n_way, k_shot, k_query, resize,
                 log_file=None, method=None):

        self.split_folder = 'dataset/tier_split'
        self.data_folder = root
        self.method = method
        self.split = mode
        self.log_file = log_file

        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.resize = resize

        # to generate self.images and self.labels
        self._read_data()
        print_log('\t\t%s set, im_num:%d, %d-way, %d-shot, %d-query, im_resize:%d' %
                  (mode, self.images.shape[0], n_way, k_shot, k_query, resize), log_file)

        self.transform = T.Compose([
            lambda ind: Image.fromarray(self.images[ind]).convert('RGB'),   # self.images[ind]: array, 0-255
            T.Resize((self.resize, self.resize)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))       # TODO: comptute this on tier-imagenet
        ])

        self.cls_num = len(np.unique(self.labels))
        self.support_sz = self.n_way * self.k_shot      # num of samples per support set
        self.query_sz = self.n_way * self.k_query       # num of samples per query set
        self.support_x_batch = []                       # support set batch
        self.query_x_batch = []                         # query set batch
        self.support_y_batch, self.query_y_batch = [], []
        self._create_batch()
        print_log('\t{}_data created!!! Not db. Length: {}'.format(self.split, len(self)), self.log_file)

    def _read_data(self):

        cache_path = os.path.join(self.data_folder, self.split)
        cache_path_labels = cache_path + "_labels.pkl"
        cache_path_images = cache_path + "_images.npz"

        if not os.path.exists(cache_path_images):
            png_pkl = cache_path_images[:-4] + '_png.pkl'
            if os.path.exists(png_pkl):
                decompress(cache_path_images, png_pkl)
            else:
                FileNotFoundError('file not exists! {}'.format(png_pkl))

        assert os.path.exists(cache_path_labels)
        assert os.path.exists(cache_path_images)

        print_log("\tRead cached labels from {}".format(cache_path_labels), self.log_file)
        with open(cache_path_labels, "rb") as f:
            data = pkl.load(f, encoding='bytes')
            self._label_specific = data[b"label_specific"]
            self._label_general = data[b"label_general"]
            self._label_specific_str = data[b"label_specific_str"]
            self._label_general_str = data[b"label_general_str"]

        self.labels_str = self._label_specific_str
        self.labels = self._label_specific

        print_log("\tRead cached images from {}".format(cache_path_images), self.log_file)
        with np.load(cache_path_images, mmap_mode="r", encoding='latin1') as data:
            self.images = data["images"]

    def _create_batch(self):
        """create batch for meta-learning in ONE episode."""
        for _ in range(len(self)):  # for each batch

            # 1. select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            support_x, query_x = [], []
            support_y, query_y = [], []

            # 2. select k_shot + k_query for each class
            for cls in selected_cls:

                selected_imgs_idx = np.random.choice(
                    np.nonzero(self.labels == cls)[0], self.k_shot + self.k_query, False)
                indexDtrain = selected_imgs_idx[:self.k_shot]     # idx for Dtrain
                indexDtest = selected_imgs_idx[self.k_shot:]      # idx for Dtest

                support_x.extend(indexDtrain)
                query_x.extend(indexDtest)
                support_y.extend(np.array([cls for _ in range(self.k_shot)]))
                query_y.extend(np.array([cls for _ in range(self.k_query)]))

            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)
            self.support_y_batch.append(support_y)
            self.query_y_batch.append(query_y)

    def __getitem__(self, index):
        """index means index of sets, 0<= index < len(self) """
        if self.method == 'gnn':
            NotImplementedError()
            # pos_cls = random.randint(0, self.n_way-1)
            # index_perm = np.random.permutation(self.n_way * self.k_shot)
            # selected_cls = np.random.choice(self.cls_num, self.n_way, False)
            #
            # labels_x = np.zeros(self.n_way, dtype='float32')
            # hidden_labels = np.zeros(self.n_way*self.k_shot+1, dtype=np.float32)
            #
            # xi = torch.FloatTensor(self.n_way*self.k_shot, 3, self.resize, self.resize).zero_()
            # labels_xi = torch.FloatTensor(self.n_way*self.k_shot, self.n_way).zero_()
            # oracles_xi = labels_xi
            #
            # # for EACH selected class and sample
            # for cls_cnt, curr_cls in enumerate(selected_cls):
            #     if cls_cnt == pos_cls:
            #         samples = self._read_im(random.sample(self.data[curr_cls], self.k_shot+1))  # why?
            #         x, labels_x[cls_cnt] = samples[0], 1
            #         samples = samples[1:]
            #     else:
            #         samples = self._read_im(random.sample(self.data[curr_cls], self.k_shot))
            #
            #     xi[index_perm[cls_cnt*self.k_shot:cls_cnt*self.k_shot+self.k_shot]] = samples
            #     # NOTE: unlabeled_extra case not implemented
            #     labels_xi[index_perm[cls_cnt*self.k_shot:cls_cnt*self.k_shot+self.k_shot], cls_cnt] = 1
            #     oracles_xi[index_perm[cls_cnt * self.k_shot:cls_cnt * self.k_shot + self.k_shot], cls_cnt] = 1
            #
            # return x, torch.from_numpy(labels_x), \
            #        xi, labels_xi, oracles_xi, hidden_labels

        else:
            support_y = np.array(self.support_y_batch[index])
            query_y = np.array(self.query_y_batch[index])

            support_x = torch.FloatTensor(self.support_sz, 3, self.resize, self.resize)
            query_x = torch.FloatTensor(self.query_sz, 3, self.resize, self.resize)
            for i, curr_ind in enumerate(self.support_x_batch[index]):
                support_x[i] = self.transform(curr_ind)

            for i, curr_ind in enumerate(self.query_x_batch[index]):
                query_x[i] = self.transform(curr_ind)

            return support_x, torch.LongTensor(torch.from_numpy(support_y)), \
                   query_x, torch.LongTensor(torch.from_numpy(query_y))

    def __len__(self):
        return int(np.floor(self.images.shape[0] * 1. / self.support_sz))


# if __name__ == '__main__':
#     # the following episode is to view one set of images via tensorboard.
#     from torchvision.utils import make_grid
#     from matplotlib import pyplot as plt
#     from tensorboardX import SummaryWriter
#     import time
#     plt.ion()
#
#     tb = SummaryWriter('runs', 'mini-imagenet')
#     mini = miniImagenet('../mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)
#
#     for i, set_ in enumerate(mini):
#         # support_x: [k_shot*n_way, 3, 84, 84]
#         support_x, support_y, query_x, _ = set_
#         support_x = make_grid(support_x, nrow=2)
#         query_x = make_grid(query_x, nrow=2)
#
#         plt.figure(1)
#         plt.imshow(support_x.transpose(2, 0).numpy())
#         plt.pause(0.5)
#         plt.figure(2)
#         plt.imshow(query_x.transpose(2, 0).numpy())
#         plt.pause(0.5)
#
#         tb.add_image('support_x', support_x)
#         tb.add_image('query_x', query_x)
#
#         time.sleep(5)
#     tb.close()

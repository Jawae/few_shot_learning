import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import numpy as np
from PIL import Image
import csv
from tools.utils import print_log


class miniImagenet(Dataset):
    """
    put mini-imagenet files as:
        root :
        |- images/*.jpg includes all images
        |- train.csv
        |- test.csv
        |- val.csv

    NOTICE:
    meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: contains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
"""

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0, log_file=None):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of images
        :param n_way:
        :param k_shot:
        :param k_query: num of query images per class
        :param resize:
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize
        self.startidx = startidx

        print_log('\t\t%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' %
                  (mode, batchsz, n_way, k_shot, k_query, resize), log_file)

        # if mode == 'train':
        # 	self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
        # 	                                     transforms.RandomResizedCrop(self.resize, scale=(0.8, 1.0)),
        # 	                                     # transforms.RandomHorizontalFlip(),
        # 	                                     # transforms.RandomVerticalFlip(),
        # 	                                     transforms.RandomRotation(15),
        # 	                                     transforms.ColorJitter(0.1, 0.1, 0.2, 0),
        # 	                                     transforms.ToTensor(),
        # 	                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # 	                                     ])
        # else:
        self.transform = T.Compose([
            lambda x: Image.open(x).convert('RGB'),
            T.Resize((self.resize, self.resize)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.path = os.path.join(root, 'images')
        csvdata = self._loadCSV(os.path.join(root, mode + '.csv'))
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, img222, ...]]
            self.img2label[k] = i + self.startidx  # {"img_name[:9]": label}

        self.cls_num = len(self.data)

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self._create_batch(self.batchsz)

    @staticmethod
    def _loadCSV(csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def _create_batch(self, batchsz):
        """
        create batch for meta-learning.
        *episode* here means batch, and it means how many sets we want to retain.
        """
        episode = batchsz
        for b in range(episode):  # for each batch

            # 1. select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                # get all images filename for current Dtrain
                support_x.append(np.array(self.data[cls])[indexDtrain].tolist())
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        """
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)

        flatten_support_x_path = \
            [os.path.join(self.path, item) for cls in self.support_x_batch[index] for item in cls]
        support_y = np.array([
            self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
            for cls in self.support_x_batch[index] for item in cls
        ])
        flatten_query_x_path = [os.path.join(self.path, item) for sublist in self.query_x_batch[index] for item in
                                sublist]
        query_y = np.array([
            self.img2label[item[:9]] for sublist in self.query_x_batch[index] for item in sublist
        ])

        for i, path in enumerate(flatten_support_x_path):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x_path):
            query_x[i] = self.transform(path)

        return support_x, torch.LongTensor(torch.from_numpy(support_y)), \
               query_x, torch.LongTensor(torch.from_numpy(query_y))

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

# if __name__ == '__main__':
# 	# the following episode is to view one set of images via tensorboard.
# 	from torchvision.utils import make_grid
# 	from matplotlib import pyplot as plt
# 	from tensorboardX import SummaryWriter
# 	import time
#
# 	plt.ion()
#
# 	tb = SummaryWriter('runs', 'mini-imagenet')
# 	mini = miniImagenet('../mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)
#
# 	for i, set_ in enumerate(mini):
# 		# support_x: [k_shot*n_way, 3, 84, 84]
# 		support_x, support_y, query_x, query_y = set_
#
# 		support_x = make_grid(support_x, nrow=2)
# 		query_x = make_grid(query_x, nrow=2)
#
# 		plt.figure(1)
# 		plt.imshow(support_x.transpose(2, 0).numpy())
# 		plt.pause(0.5)
# 		plt.figure(2)
# 		plt.imshow(query_x.transpose(2, 0).numpy())
# 		plt.pause(0.5)
#
# 		tb.add_image('support_x', support_x)
# 		tb.add_image('query_x', query_x)
#
# 		time.sleep(5)
#
# 	tb.close()

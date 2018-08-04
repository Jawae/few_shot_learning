from torch.utils.data import DataLoader
# print(sys.path)
from dataset.miniImagenet import miniImagenet
from dataset.omniglot import OmniglotDataset
from tools.utils import print_log
import numpy as np
from nips17_proto.batch_sampler import PrototypicalBatchSampler


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.k_shot + opt.k_query
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(
        labels=labels, classes_per_it=classes_per_it,
        num_samples=num_samples, iterations=opt.iterations)


def data_loader(opts):

    train_db, val_db, test_db, trainval_db = [], [], [], []
    print_log('\nPreparing datasets: [{:s}] ...'.format(opts.dataset), opts.log_file)

    if opts.dataset == 'mini-imagenet':

        train_data = miniImagenet('dataset/miniImagenet/', mode='train',
                                  n_way=opts.n_way, k_shot=opts.k_shot, k_query=opts.k_query,
                                  batchsz=opts.meta_batchsz_train, resize=opts.im_size, log_file=opts.log_file)
        train_db = DataLoader(train_data, opts.batch_sz, shuffle=True, num_workers=8, pin_memory=True)

        val_data = miniImagenet('dataset/miniImagenet/', mode='val',
                                n_way=opts.n_way, k_shot=opts.k_shot, k_query=opts.k_query,
                                batchsz=opts.meta_batchsz_test, resize=opts.im_size, log_file=opts.log_file)
        val_db = DataLoader(val_data, opts.batch_sz, shuffle=True, num_workers=2, pin_memory=True)

    elif opts.dataset == 'omniglot':

        print_log('\ntrain data ...', opts.log_file)
        train_data = OmniglotDataset(mode='train', root='dataset/omniglot')    # mode: train/test/val/trainval
        n_classes = len(np.unique(train_data.y))
        if n_classes < opts.classes_per_it_tr or n_classes < opts.classes_per_it_val:
            raise Exception('There are not enough classes in the dataset in order '
                            'to satisfy the chosen classes_per_it. Decrease the '
                            'classes_per_it_{tr/val} option and try again.')
        train_db = DataLoader(train_data, batch_sampler=init_sampler(opts, train_data.y, 'train'))

        print_log('\nval data ...', opts.log_file)
        val_data = OmniglotDataset(mode='val', root='dataset/omniglot')
        val_db = DataLoader(val_data, batch_sampler=init_sampler(opts, val_data.y, 'val'))

        print_log('\ntest data ...', opts.log_file)
        test_data = OmniglotDataset(mode='test', root='dataset/omniglot')
        test_db = DataLoader(test_data, batch_sampler=init_sampler(opts, test_data.y, 'test'))

    return train_db, val_db, test_db, trainval_db

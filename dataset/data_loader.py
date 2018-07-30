from torch.utils.data import DataLoader
# print(sys.path)
from dataset.miniImagenet import miniImagenet
from tools.utils import print_log


def data_loader(opts):

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

    else:
        NotImplementedError()

    return train_db, val_db
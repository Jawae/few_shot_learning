from __future__ import print_function
import argparse
import sys

import torch.nn.functional as F
import torch.optim as optim

from basic_opt import *
from models import EmbeddingImagenet, EmbeddingOmniglot, MetricNN

sys.path.append(os.getcwd())
from dataset.data_loader import data_loader
from torch.optim.lr_scheduler import MultiStepLR
from tools.utils import print_log


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='omniglot')
    # parser.add_argument('-dataset', type=str, default='mini-imagenet')
    parser.add_argument('-n_way', type=int, default=5)              # train_N_way
    parser.add_argument('-k_shot', type=int, default=2)             # train_N_shots
    parser.add_argument('-k_query', type=int, default=1)
    parser.add_argument('-gpu_id', type=int, nargs='+', default=0)

    parser.add_argument('-im_size', type=int, default=224)
    parser.add_argument('-meta_batchsz_train', type=int, default=10000)
    parser.add_argument('-meta_batchsz_test', type=int, default=200)

    parser.add_argument('-unlabeled_extra', type=int, default=0, help='Number of shots when training')
    parser.add_argument('-metric_network', type=str, default='gnn_iclr_nl', help='gnn_iclr_nl' + 'gnn_iclr_active')
    parser.add_argument('-active_random', type=int, default=0, help='random active ? ')
    return parser


# PARAMS
opts = get_basic_parser(get_parser()).parse_args()
opts.method = 'gnn'
setup(opts)

# CREATE MODEL
if opts.dataset == 'omniglot':
    enc_nn = EmbeddingOmniglot(opts, 64).to(opts.device)
elif opts.dataset == 'mini-imagenet':
    enc_nn = EmbeddingImagenet(opts, 128).to(opts.device)
metric_nn = MetricNN(opts, emb_size=enc_nn.emb_size).to(opts.device)

# RESUME (fixme with appropriate epoch and iter)
if os.path.exists(opts.model_file) and os.path.exists(opts.model_file2):
    print_log('loading previous best checkpoint [{}] ...'.format(opts.model_file), opts.log_file)
    enc_nn.load_state_dict(torch.load(opts.model_file))
    print_log('loading previous best checkpoint [{}] ...'.format(opts.model_file2), opts.log_file)
    metric_nn.load_state_dict(torch.load(opts.model_file2))

if opts.multi_gpu:
    print_log('Wrapping network into multi-gpu mode ...', opts.log_file)
    enc_nn = torch.nn.DataParallel(enc_nn)
    metric_nn = torch.nn.DataParallel(metric_nn)


# PREPARE DATA
# train_loader = generator.Generator(args.dataset_root, args, partition='train', dataset=args.dataset)
# io.cprint('Batch size: '+str(args.batch_size))
train_db, val_db, _, _ = data_loader(opts)

# MISC
# weight_decay = 0
# if args.dataset == 'mini_imagenet':
#     print('Weight decay '+str(1e-6))
#     weight_decay = 1e-6
opt_enc_nn = optim.Adam(enc_nn.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
opt_metric_nn = optim.Adam(metric_nn.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
scheduler_enc = MultiStepLR(opt_enc_nn, milestones=opts.scheduler, gamma=opts.lr_scheduler_gamma)
scheduler_metric = MultiStepLR(opt_metric_nn, milestones=opts.scheduler, gamma=opts.lr_scheduler_gamma)


# PIPELINE
counter = 0
total_loss = 0
val_acc, val_acc_aux = 0, 0
test_acc = 0

total_ep, total_iter = opts.nep, len(train_db)

for epoch in range(opts.nep):

    old_lr = opt_enc_nn.param_groups[0]['lr']   # assume these two opt have the same schedule
    scheduler_enc.step()
    scheduler_metric.step()
    new_lr = opt_enc_nn.param_groups[0]['lr']

    if epoch == 0:
        print_log('\tInitial lr is {:.8f}\n'.format(old_lr), opts.log_file)
    if new_lr != old_lr:
        print_log('\tLR changes from {:.8f} to {:.8f} at epoch {:d}\n'.format(old_lr, new_lr, epoch), opts.log_file)

    for step, batch in enumerate(train_db):

        # TODO: get data here
        # data = train_loader.get_task_batch(batch_size=args.batch_size, n_way=args.train_N_way,
        #                                    unlabeled_extra=args.unlabeled_extra, num_shots=args.train_N_shots,
        #                                    cuda=args.cuda, variable=True, device=device)
        # [batch_x, label_x, _, _, batches_xi, labels_yi, oracles_yi, hidden_labels] = data

        # Compute embedding from x and xi_s
        z = enc_nn(batch_x)[-1]
        zi_s = [enc_nn(batch_xi)[-1] for batch_xi in batches_xi]

        # Compute metric from embeddings
        out_metric, out_logits = metric_nn(inputs=[z, zi_s, labels_yi, oracles_yi, hidden_labels])
        logsoft_prob = F.log_softmax(out_logits)

        # Loss
        # fixme why to cpu again?
        # label_x_numpy = label_x.cpu().data.numpy()
        # formatted_label_x = np.argmax(label_x_numpy, axis=1)
        # formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
        formatted_label_x = torch.argmax(label_x, dim=1)
        loss_d_metric = F.nll_loss(logsoft_prob, formatted_label_x)

        opt_enc_nn.zero_grad()
        opt_metric_nn.zero_grad()

        loss_d_metric.backward()

        opt_enc_nn.step()
        opt_metric_nn.step()

        # SHOW TRAIN LOSS
        counter += 1
        total_loss += loss_d_metric.item()
        if step % opts.iter_vis_loss == 0 or step == len(train_db)-1:
            print_log(opts.loss_vis_str.format(total_ep, epoch, total_iter, step, total_loss/counter), opts.log_file)
            counter = 0
            total_loss = 0

        # VALIDATION SET
        if step % opts.iter_do_val == 0:
            enc_nn.eval()
            metric_nn.eval()

            with torch.no_grad():
                for j, batch_test in enumerate(val_db):
                    NotImplementedError()
                    # TODO

            enc_nn.train()
            metric_nn.train()
            # show results, save model below




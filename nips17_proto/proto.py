# coding=utf-8
from tqdm import tqdm
import sys
import argparse
from torch import optim

from basic_opt import *
from prototypical_loss import prototypical_loss as loss_fn
from protonet import ProtoNet
sys.path.append(os.getcwd())
from dataset.data_loader import data_loader
from torch.optim.lr_scheduler import MultiStepLR, StepLR


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='omniglot')
    # default='omniglot')
    # used in data_loader.py for omniglot
    parser.add_argument('-classes_per_it_tr', type=int, default=60)     # just the N-way
    parser.add_argument('-classes_per_it_val', type=int, default=5)

    # used in 'init_sampler' method
    parser.add_argument('-k_shot', type=int, default=5)                 # old name: num_support_tr
    parser.add_argument('-k_query', type=int, default=5)                # old name: num_query_tr
    parser.add_argument('-num_support_val', type=int, default=5)        # just the k_shot for validation
    parser.add_argument('-num_query_val', type=int, default=15)         # just the k_query for validation

    parser.add_argument('-gpu_id', type=int, nargs='+', default=0)
    # parser.add_argument('-im_size', type=int, default=224)
    # parser.add_argument('-network', type=str, default='resnet18')
    # parser.add_argument('-meta_batchsz_train', type=int, default=10000)
    # parser.add_argument('-meta_batchsz_test', type=int, default=200)
    parser.add_argument('-distance', type=str, help='cosine or euclidean', default='euclidean')
    return parser


# PARAMS
opts = get_basic_parser(get_parser()).parse_args()
opts.method = 'proto'
setup(opts)

# CREATE MODEL
net = ProtoNet().to(opts.device)

# RESUME (fixme with appropriate epoch and iter)
if os.path.exists(opts.model_file):
    print_log('loading previous best checkpoint [{}] ...'.format(opts.model_file), opts.log_file)
    net.load_state_dict(torch.load(opts.model_file))

if opts.multi_gpu:
    print_log('Wrapping network into multi-gpu mode ...', opts.log_file)
    net = torch.nn.DataParallel(net)

# PREPARE DATA
train_db, val_db, test_db, _ = data_loader(opts)

# MISC
# TODO: original repo don't have weight decay
optimizer = optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
# scheduler = MultiStepLR(optimizer, milestones=opts.scheduler, gamma=opts.lr_scheduler_gamma)
scheduler = StepLR(optimizer, gamma=opts.lr_scheduler_gamma, step_size=opts.lr_scheduler_step)


# PIPELINE
if val_db is None:
    best_state = None
train_loss, train_acc, val_loss, val_acc, best_acc = [], [], [], [], 0

for epoch in range(opts.nep):

    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    new_lr = optimizer.param_groups[0]['lr']

    if epoch == 0:
        print_log('\tInitial lr is {:.8f}\n'.format(old_lr), opts.log_file)
    if new_lr != old_lr:
        print_log('\tLR changes from {:.8f} to {:.8f} at epoch {:d}\n'.format(old_lr, new_lr, epoch), opts.log_file)

    tr_iter = iter(train_db)

    for batch in tqdm(tr_iter):

        net.train()
        x, y = batch[0].to(opts.device), batch[1].to(opts.device)
        # TODO use k_query or not?
        loss, acc = loss_fn(net(x), target=y, n_support=opts.k_shot, distance=opts.distance)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        train_acc.append(acc.item())
    # ONE EPOCH ENDS

    avg_loss = np.mean(train_loss[-opts.iterations:])  # TODO: why need iterations?
    avg_acc = np.mean(train_acc[-opts.iterations:])
    print_log('Avg Train Loss: {:.5f}, Avg Train Acc: {:.5f}'.format(avg_loss, avg_acc), opts.log_file)

    if val_db is None:
        continue
    val_iter = iter(val_db)
    net.eval()
    for batch in val_iter:
        x, y = batch[0].to(opts.device), batch[0].to(opts.device)
        loss, acc = loss_fn(net(x), target=y, n_support=opts.num_support_val, distance=opts.distance)
        val_loss.append(loss.item())
        val_acc.append(acc.item())
    avg_loss = np.mean(val_loss[-opts.iterations:])
    avg_acc = np.mean(val_acc[-opts.iterations:])
    postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {:.5f})'.format(best_acc)
    print_log('Avg Val Loss: {:.5f}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix), opts.log_file)
    if avg_acc >= best_acc:
        best_acc = avg_acc
        if opts.multi_gpu:
            torch.save(net.module.state_dict(), opts.model_file)
        else:
            torch.save(net.state_dict(), opts.model_file)
        print_log('[epoch {} / iter {}] best model saved to: {}'.format(
            epoch, len(train_db), opts.model_file), file=opts.log_file)
        best_acc = avg_acc
        best_state = net.state_dict()
# TRAINING ENDS

if best_state is not None:
    net.load_state_dict(best_state)   # fixme when multi gpu, net.module()
    print_log('Testing with best model ...', opts.log_file)
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_db)
        for batch in test_iter:
            x, y = batch
            x, y = batch[0].to(opts.device), batch[1].to(opts.device)
            _, acc = loss_fn(net(x), target=y, n_support=opts.k_shot, distance=opts.distance)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print_log('Test Acc: {:.6f}'.format(avg_acc), opts.log_file)


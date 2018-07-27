import argparse
import numpy as np
from torch import optim
from network import Relation
from utils import make_imgs

from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from datetime import datetime
from basic_opt import *
import sys
sys.path.append(os.getcwd())
from dataset.data_loader import data_loader
from torch.optim.lr_scheduler import MultiStepLR


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n_way', type=int, default=5)
    parser.add_argument('-k_shot', type=int, default=2)
    parser.add_argument('-k_query', type=int, default=1)
    parser.add_argument('-im_size', type=int, default=224)
    parser.add_argument('-meta_batchsz_train', type=int, default=500)   # 10000
    parser.add_argument('-meta_batchsz_test', type=int, default=200)
    return parser


# PARAMS
opts = get_basic_parser(get_parser()).parse_args()
opts.method = 'relation'
setup(opts)

# CREATE MODEL
net = Relation(opts).to(opts.device)

# RESUME (fixme with appropriate epoch and iter)
if os.path.exists(opts.model_file):
    print('loading previous best checkpoint ...', opts.model_file)
    net.load_state_dict(torch.load(opts.model_file))

if opts.multi_gpu:
    print('Wrapping network into multi-gpu mode ...')
    net = torch.nn.DataParallel(net)

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Total params in the network:', params)

# PREPARE DATA
train_db, val_db = data_loader(opts)

# MISC
optimizer = optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=opts.scheduler, gamma=opts.lr_scheduler_step)

if opts.use_tensorboard:
    tb = SummaryWriter(opts.tb_folder, str(datetime.now()))

# PIPELINE
best_accuracy = 0
print('\nPipeline starts now !!!')

old_lr = optimizer.param_groups[0]['initial_lr']

for epoch in range(opts.nep):

    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    new_lr = optimizer.param_groups[0]['lr']

    if epoch == 0:
        print('\tInitial lr is {:.8f}'.format(old_lr))
    if new_lr != old_lr:
        print('\tLR changes from {:.8f} to {:.8f} at epoch {:d}'.format(old_lr, new_lr, epoch))

    for step, batch in enumerate(train_db):

        net.train()
        support_x, support_y, query_x, query_y = \
            batch[0].to(opts.device), batch[1].to(opts.device), \
            batch[2].to(opts.device), batch[3].to(opts.device)

        loss = net(support_x, support_y, query_x, query_y)
        # print(loss.size())
        loss = loss.mean()      # multi-gpu
        # print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # SHOW TRAIN LOSS
        if step % opts.iter_vis_loss == 0 or step == len(train_db)-1:
            if opts.use_tensorboard:
                tb.add_scalar('loss', loss.item())
            print(opts.loss_vis_str.format(epoch, step, loss.item()))

        # VALIDATION SET
        if step % opts.iter_do_val == 0:

            total_correct, total_num, display_onebatch = 0, 0, False

            for j, batch_test in enumerate(val_db):

                net.eval()
                support_x, support_y, query_x, query_y = \
                    batch_test[0].to(opts.device), batch_test[1].to(opts.device), \
                    batch_test[2].to(opts.device), batch_test[3].to(opts.device)

                pred, correct = net(support_x, support_y, query_x, query_y, False)
                correct = correct.sum()     # multi-gpu support
                total_correct += correct.item()
                total_num += query_y.size(0) * query_y.size(1)

                if not display_onebatch:
                    display_onebatch = True  # only display once
                    all_img, max_width = \
                        make_imgs(n_way, k_shot, k_query,
                                  support_x.size(0), support_x, support_y, query_x, query_y,
                                  pred, device)
                    all_img = make_grid(all_img, nrow=max_width)
                    tb.add_image('result batch', all_img)

            accuracy = total_correct / total_num
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if opts.multi_gpu:
                    torch.save(net.module.state_dict(), opts.model_file)
                else:
                    torch.save(net.state_dict(), opts.model_file)
                print('best model saved to:', opts.model_file)

            tb.add_scalar('accuracy', accuracy)
            print('<<<<>>>>accuracy:', accuracy, 'best accuracy:', best_accuracy)


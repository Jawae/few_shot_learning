import torch
import os
import numpy as np
from torch import optim
from torch.autograd import Variable
from compare import Compare
from utils import make_imgs

from dataset.miniImagenet import miniImagenet
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from datetime import datetime

n_way = 5
k_shot = 1
k_query = 1  # query num per class
batchsz = 1
gpu_id = 0

device = 'cuda:{}'.format(gpu_id) if torch.cuda.is_available() and gpu_id > -1 else 'cpu'

# print('To run on single GPU, change device_ids=[0] and downsize batch size! \nmkdir ckpt if not exists!')
# net = torch.nn.DataParallel(Compare(n_way, k_shot), device_ids=[0]).cuda()
net = Compare(n_way, k_shot, gpu_id=gpu_id).to(device)
# print(net)
if not os.path.exists('output/ckpt'):
    os.makedirs('output/ckpt')

mdl_file = 'output/ckpt/compare%d%d.mdl' % (n_way, k_shot)
if os.path.exists(mdl_file):
    print('load checkpoint ...', mdl_file)
    net.load_state_dict(torch.load(mdl_file))

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('total params:', params)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
tb = SummaryWriter('runs', str(datetime.now()))

mini = miniImagenet('dataset/miniImagenet/', mode='train',
                    n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=10000, resize=224)
db = DataLoader(mini, batchsz, shuffle=True, num_workers=8, pin_memory=True)
mini_val = miniImagenet('dataset/miniImagenet/', mode='val',
                        n_way=n_way, k_shot=k_shot, k_query=k_query, batchsz=200, resize=224)
db_val = DataLoader(mini_val, batchsz, shuffle=True, num_workers=2, pin_memory=True)

best_accuracy = 0
for epoch in range(1000):

    for step, batch in enumerate(db):

        support_x, support_y, query_x, query_y = batch
        support_x, support_y, query_x, query_y = \
            support_x.to(device), support_y.to(device), query_x.to(device), query_y.to(device)

        net.train()
        loss = net(support_x, support_y, query_x, query_y)
        loss = loss.mean()  # Multi-GPU support

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # SHOW TRAIN LOSS
        if step % 15 == 0 and step != 0:
            tb.add_scalar('loss', loss.item())
            print('%d-way %d-shot %d batch> epoch:%d step:%d, loss:%f' %
                  (n_way, k_shot, batchsz, epoch, step, loss.cpu().data[0]))

        # VALIDATION SET
        total_val_loss = 0
        if step % 1 == 0:

            total_correct, total_num, display_onebatch = 0, 0, False

            for j, batch_test in enumerate(db_val):

                support_x, support_y, query_x, query_y = batch_test
                support_x, support_y, query_x, query_y = \
                    support_x.to(device), support_y.to(device), query_x.to(device), query_y.to(device)

                net.eval()
                pred, correct = net(support_x, support_y, query_x, query_y, False)
                correct = correct.sum()     # multi-gpu support
                total_correct += correct.item()
                total_num += query_y.size(0) * query_y.size(1)

                if not display_onebatch:
                    display_onebatch = True  # only display once
                    all_img, max_width = make_imgs(
                        n_way, k_shot, k_query,
                        support_x.size(0), support_x, support_y, query_x, query_y, pred,
                        device
                    )
                    all_img = make_grid(all_img, nrow=max_width)
                    tb.add_image('result batch', all_img)

            accuracy = total_correct / total_num
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(net.state_dict(), mdl_file)
                print('saved to checkpoint:', mdl_file)

            tb.add_scalar('accuracy', accuracy)
            print('<<<<>>>>accuracy:', accuracy, 'best accuracy:', best_accuracy)


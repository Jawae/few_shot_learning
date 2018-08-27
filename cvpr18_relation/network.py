import os
import math
import torch
from torch import nn
from torch.nn import functional as F
# from repnet import repnet_deep, Bottleneck
import sys
sys.path.append(os.getcwd())
from tools.utils import print_log
from torch.utils import model_zoo


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=64):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


def repnet_deep(pretrained=False, **kwargs):
    """Constructs a ResNet-Mini-Imagenet model"""
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }
    # TODO: for now only supports resnet18
    if kwargs['structure'] == 'resnet18':
        model = ResNet(Bottleneck, [3, 4, 6])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[kwargs['structure']]), strict=False)
    return model


class Relation(nn.Module):
    """
    repnet => feature concat => layer4 & layer5 & avg pooling => fc => sigmoid
    """
    def __init__(self, opts):
        super(Relation, self).__init__()
        # self.opts = opts
        self.n_way = opts.n_way
        self.k_shot = opts.k_shot
        self.device = opts.device

        print_log('\nBuilding up models ...', opts.log_file)
        self.repnet = repnet_deep(False, structure=opts.network)
        random_input = torch.rand(2, 3, opts.im_size, opts.im_size)
        repnet_out = self.repnet(random_input)

        repnet_sz = repnet_out.size()
        self.c = repnet_sz[1]
        self.d = repnet_sz[2]
        # this is the input channels of layer4 and layer5
        self.inplanes = 2 * self.c
        assert repnet_sz[2] == repnet_sz[3]
        print_log('\t\trepnet sz: {}'.format(repnet_sz), opts.log_file)

        # after the relation module
        self.layer4 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 64, 3, stride=2)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        combine = torch.stack([repnet_out, repnet_out], dim=1).view(
            repnet_out.size(0), -1, repnet_out.size(2), repnet_out.size(3))
        out = self.layer5(self.layer4(combine))
        print_log('\t\tafter layer5 sz: {}'.format(out.size()), opts.log_file)
        self.pool_size = out.size(2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, support_x, support_y, query_x, query_y, train=True):
        """
        :param support_x: 	[b, setsz, c_, h, w]
        :param support_y: 	[b, setsz]
        :param query_x:   	[b, querysz, c_, h, w]
        :param query_y:   	[b, querysz]
        :param train:	 	train or not
        :return:			loss or prediction
        """
        # self.setsz = self.n_way * self.k_shot  # num of samples per set
        # self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation

        batchsz, setsz, c_, h, w = support_x.size()
        querysz = query_x.size(1)
        c, d = self.c, self.d   # todo: can c and d be self-determined?

        support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
        query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

        # [b, setsz, c, d, d] => [b, 1, setsz, c, d, d] => [b, querysz, setsz, c, d, d]
        support_xf = support_xf.unsqueeze(1).expand(-1, querysz, -1, -1, -1, -1)
        # [b, querysz, c, d, d] => [b, querysz, 1, c, d, d] => [b, querysz, setsz, c, d, d]
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, setsz, -1, -1, -1)

        # cat: 2 x [b, querysz, setsz, c, d, d] => [b, querysz, setsz, 2c, d, d]
        comb = torch.cat([support_xf, query_xf], dim=3)
        comb = self.layer5(self.layer4(comb.view(batchsz * querysz * setsz, 2 * c, d, d)))
        comb = F.avg_pool2d(comb, self.pool_size)

        # push to Linear layer
        # [b * querysz * setsz, 256] => [b * querysz * setsz, 1] => [b, querysz, setsz, 1]
        # score: [b, querysz, setsz]
        score = self.fc(comb.view(batchsz * querysz * setsz, -1)).view(batchsz, querysz, setsz, 1).squeeze(3)

        if train:
            # build the label
            # [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
            support_y_expand = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
            # [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
            query_y_expand = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
            # eq: [b, querysz, setsz] vs [b, querysz, setsz]
            # convert byte tensor to float tensor
            # label: [b, querysz, setsz]
            label = torch.eq(support_y_expand, query_y_expand).float()

            loss = torch.pow(label - score, 2).sum() / batchsz
            loss = loss.unsqueeze(0)
            # print(loss.size())
            # print(loss.item())
            return loss
        else:
            # TEST
            temp = score.view(score.size(0), score.size(1), self.n_way, self.k_shot)
            # pred_ind: b, querysz (n_way x self.k_query)
            pred_ind = temp.sum(dim=-1).argmax(dim=-1)  # TODO: replace sum with avg or other

            support_y_neat = support_y[:, ::self.k_shot]  # b, n_way
            pred = torch.stack([support_y_neat[b, ind] for b, query in enumerate(pred_ind) for ind in query])
            pred = pred.view(score.size(0), -1)

            correct = torch.eq(pred, query_y).sum()
            correct = correct.unsqueeze(0)
            # print('pred size {}'.format(pred.size()))
            # print('correct size {}'.format(correct.size()))
            return pred, correct

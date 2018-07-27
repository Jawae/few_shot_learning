import os
import torch
from torch import nn
from torch.nn import functional as F
from repnet import repnet_deep, Bottleneck
import sys
# print(os.getcwd())  # root dir
sys.path.append(os.getcwd())
from utils.utils import print_log


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
		# resnet-18 model; why they called repnet?
		# RE: might be representation network
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
		c, d = self.c, self.d

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
			pred_ind = temp.sum(dim=-1).argmax(dim=-1)   # TODO: replace sum with avg or other

			support_y_neat = support_y[:, ::self.k_shot]   # b, n_way
			pred = torch.stack([support_y_neat[b, ind] for b, query in enumerate(pred_ind) for ind in query])
			pred = pred.view(score.size(0), -1)

			correct = torch.eq(pred, query_y).sum()
			correct = correct.unsqueeze(0)
			# print('pred size {}'.format(pred.size()))
			# print('correct size {}'.format(correct.size()))
			return pred, correct

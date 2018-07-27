import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from repnet import repnet_deep, Bottleneck


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

		print('\nBuilding up models ...')
		self.repnet = repnet_deep(False)    # resnet-18 model; why they called repnet?
		repnet_sz = self.repnet(torch.rand(2, 3, opts.im_size, opts.im_size)).size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		# this is the input channels of layer4 and layer5
		self.inplanes = 2 * self.c
		assert repnet_sz[2] == repnet_sz[3]
		print('\t\trepnet sz:', repnet_sz)

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
		:return:
		"""
		batchsz, setsz, c_, h, w = support_x.size()
		querysz = query_x.size(1)
		c, d = self.c, self.d

		support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
		query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

		# [b, setsz, c, d, d] => [b, 1, setsz, c, d, d] => [b, querysz, setsz, c, d, d]
		support_xf = support_xf.unsqueeze(1).expand(-1, querysz, -1, -1, -1, -1)
		# [b, querysz, c, d, d] => [b, querysz, 1, c, d, d] => [b, querysz, setsz, c, d, d]
		query_xf = query_xf.unsqueeze(2).expand(-1, -1, setsz, -1, -1, -1)
		# cat: [b, querysz, setsz, c, d, d] => [b, querysz, setsz, 2c, d, d]
		comb = torch.cat([support_xf, query_xf], dim=3)

		comb = self.layer5(self.layer4(comb.view(batchsz * querysz * setsz, 2 * c, d, d)))
		comb = F.avg_pool2d(comb, 3)   # TODO: check with different input image sizes
		# push to Linear layer
		# [b * querysz * setsz, 256] => [b * querysz * setsz, 1] => [b, querysz, setsz, 1] => [b, querysz, setsz]
		score = self.fc(comb.view(batchsz * querysz * setsz, -1)).view(batchsz, querysz, setsz, 1).squeeze(3)

		# build the label
		# [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
		support_yf = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
		# [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
		query_yf = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
		# eq: [b, querysz, setsz] => [b, querysz, setsz] and convert byte tensor to float tensor
		label = torch.eq(support_yf, query_yf).float()

		# score: [b, querysz, setsz]
		# label: [b, querysz, setsz]
		if train:
			loss = torch.pow(label - score, 2).sum() / batchsz
			loss = loss.unsqueeze(0)
			# print(loss.size())
			# print(loss.item())
			return loss
		else:
			# [b, querysz, setsz]
			rn_score_np = score.cpu().data.numpy()
			# [b, setsz]
			support_y_np = support_y.cpu().data.numpy()
			pred = []

			for i, batch in enumerate(rn_score_np):
				for j, query in enumerate(batch):
					# query: [setsz]
					sim = []  # [n_way]
					for way in range(self.n_way):
						sim.append(np.sum(query[way * self.k_shot: (way + 1) * self.k_shot]))
					idx = np.array(sim).argmax()
					pred.append(support_y_np[i, idx * self.k_shot])
			# pred: [b, querysz]
			pred = torch.from_numpy(np.array(pred).reshape((batchsz, querysz))).to(self.device)
			correct = torch.eq(pred, query_y).sum()
			correct = correct.unsqueeze(0)
			# print('pred size {}'.format(pred.size()))
			# print('correct size {}'.format(correct.size()))
			return pred, correct

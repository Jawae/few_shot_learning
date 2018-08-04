import os
import torch
import numpy as np


def remove(file_name):
    try:
        os.remove(file_name)
    except:
        pass


def print_log(msg, file=None, init=False, additional_file=None, quiet_termi=False):

    if not quiet_termi:
        print(msg)
    if file is None:
        pass
    else:
        if init:
            remove(file)
        with open(file, 'a') as log_file:
            log_file.write('%s\n' % msg)

        if additional_file is not None:
            # TODO (low): a little buggy here: no removal of previous additional_file
            with open(additional_file, 'a') as addition_log:
                addition_log.write('%s\n' % msg)


def im_map_back(im, std, mean):
    im = im * torch.FloatTensor(list(std)).view(1, 3, 1, 1) + \
         torch.FloatTensor(list(mean)).view(1, 3, 1, 1)
    return im


def show_result(opts, support_x, support_y, query_x, query_y, query_pred,
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """show result of one dimension (cvpr 2018 relation paper)"""
    n_way = opts.n_way
    k_shot = opts.k_shot
    n_query_per_cls = opts.k_query
    batchsz = support_x.size(0)
    device = opts.device

    # randomly select one batch
    batchidx = np.random.randint(batchsz)
    max_width = (k_shot + n_query_per_cls * 2 + 4)   # hyli: what's this?
    # de-normalize
    img_support = support_x[batchidx].clone()
    img_query = query_x[batchidx].clone()

    img_support = im_map_back(img_support, std, mean)
    img_query = im_map_back(img_query, std, mean)

    # TODO (hyli): no idea what's going on here; check later
    label = support_y[batchidx]                                             # [setsz]
    label, indices = torch.sort(label, dim=0)
    img_support = torch.index_select(img_support, dim=0, index=indices)     # [setsz, c, h, w]
    all_img = torch.zeros(max_width*n_way, *img_support[0].size())          # [max_width*n_way, c, h, w]

    for row in range(n_way):  # for each row
        # [0, k_shot)
        for pos in range(k_shot):  # copy the first k_shot
            all_img[row * max_width + pos] = img_support[row * k_shot + pos].data

        # now set the pred imgs
        # [k_shot+1, max_width - n_query_per_cls -1]
        pos = k_shot + 1  # pointer to empty buff
        for idx, img in enumerate(img_query):
            # search all imgs in pred that match current row id: label[row*k_shot]
            if torch.equal(query_pred[batchidx][idx], label[row * k_shot]):  # if pred it match current id
                if pos == max_width - n_query_per_cls:  # overwrite the last column
                    pos -= 1
                all_img[row * max_width + pos] = img.data  # copy img
                pos += 1

        # set the last several column as the right img
        #  [max_width - n_query_per_cls, max_width)
        pos = max_width - n_query_per_cls
        for idx, img in enumerate(img_query):  # search all imgs in pred that match current row id: label[row*k_shot]
            if torch.equal(query_y[batchidx][idx], label[row * k_shot]):  # if query_y id match current id
                if pos == max_width:  # overwrite the last column
                    pos -= 1
                all_img[row * max_width + pos] = img.data  # copy img
                pos += 1

    print('label for support:', label.data.cpu().numpy().tolist())
    print('label for query  :', query_y.data[batchidx].cpu().numpy())
    print('label for pred   :', query_pred.data[batchidx].cpu().numpy())

    return all_img, max_width

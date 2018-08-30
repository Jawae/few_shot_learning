# coding=utf-8
import os
import torch
import datetime
import numpy as np
from tools.utils import print_log


def get_basic_parser(parser):
    """universal arguments across different algorithms"""
    parser.add_argument('-root',
                        type=str,
                        default='output')
    # output folder: output/METHOD/DATASET/exp_name
    parser.add_argument('-exp_name', type=str, default='default')

    # TRAINING STATS
    # these numbers might differ on various datasets
    parser.add_argument('-nep',
                        type=int,
                        help='number of epochs to train for',
                        default=500)   # 100 for proto_net

    parser.add_argument('-lr',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)
    parser.add_argument('-scheduler', type=list, default=[300, 400])

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)
    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-batch_sz', type=int, default=2)
    parser.add_argument('-weight_decay', type=float, default=0.0005)

    # MISC
    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)
    parser.add_argument('-gpu_id', type=int, nargs='+', default=0)

    # TODO (low): debug mode smaller interval
    parser.add_argument('-iter_vis_loss', type=int, default=25)
    parser.add_argument('-iter_do_val', type=int, default=2000)

    # TODO: visualization
    parser.add_argument('-use_tensorboard', action='store_true')
    parser.add_argument('-use_visdom', action='store_true')
    return parser


def setup(opt2):

    np.random.seed(opt2.manual_seed)
    torch.manual_seed(opt2.manual_seed)
    torch.cuda.manual_seed(opt2.manual_seed)

    # opt2.__dict__.update(opt1.__dict__)i
    # opt2.gpu_id = opt2.device_id
    opt2.output_folder = os.path.join(opt2.root, opt2.method, opt2.dataset, opt2.exp_name)
    if not os.path.exists(opt2.output_folder):
        os.makedirs(opt2.output_folder)

    # sanity check
    if not hasattr(opt2, 'n_way'):
        opt2.n_way = opt2.classes_per_it_tr

    if opt2.method == 'gnn':
        opt2.model_file = os.path.join(
            opt2.output_folder, '{:d}_way_{:d}_shot_enc_nn.hyli'.format(opt2.n_way, opt2.k_shot))
        opt2.model_file2 = os.path.join(
            opt2.output_folder, '{:d}_way_{:d}_shot_metric_nn.hyli'.format(opt2.n_way, opt2.k_shot))

        opt2.train_N_way = opt2.n_way
        opt2.test_N_way = opt2.train_N_way
        opt2.train_N_shots = opt2.k_shot
        opt2.test_N_shots = opt2.train_N_shots
    else:
        opt2.model_file = os.path.join(
            opt2.output_folder, '{:d}_way_{:d}_shot.hyli'.format(opt2.n_way, opt2.k_shot))

    prefix = '[{:s}]'.format(opt2.model_file.replace(opt2.root+'/', '').replace('.hyli', ''))
    opt2.loss_vis_str = prefix + ' [ep({}) {:04d} / iter({}) {:06d}] loss: {:.4f}'

    if opt2.use_tensorboard:
        opt2.tb_folder = os.path.join(opt2.output_folder, 'runs')
        if not os.path.exists(opt2.tb_folder):
            os.makedirs(opt2.tb_folder)

    opt2.log_file = os.path.join(opt2.output_folder, 'training_dynamic.txt')
    now = datetime.datetime.now()
    print_log('\nStart timestamp: {:%Y%m%dT%H%M}'.format(now), file=opt2.log_file, init=True)

    if isinstance(opt2.gpu_id, int):
        opt2.gpu_id = [opt2.gpu_id]   # int -> list

    # Detect if cuda is there
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opt2.gpu_id[0] == -1 and device == 'cuda':
        print_log('you have cuda and yet gpu_id is set to -1; launching GPU anyway (use gpu 0) ...',
                  file=opt2.log_file)
        opt2.gpu_id[0] = 0
    elif opt2.gpu_id[0] > -1 and device == 'cpu':
        print_log('you have cpu only and yet gpu_id is set above -1; launching CPU anyway ...',
                  file=opt2.log_file)
        opt2.gpu_id[0] = -1
    multi_gpu = True if len(opt2.gpu_id) > 1 and device == 'cuda' else False
    if device == 'cuda':
        print_log('\nGPU mode, gpu_ids: {}'.format(opt2.gpu_id), file=opt2.log_file)
    else:
        print_log('\nCPU mode', file=opt2.log_file)

    opt2.multi_gpu = multi_gpu
    opt2.device = device

    # write params to log file
    print_log('\nWriting params to log: {}'.format(opt2.log_file), file=opt2.log_file)
    temp = opt2.__dict__
    for k in sorted(temp):
        print_log('\t{:s}:\t\t{}'.format(k, temp[k]), file=opt2.log_file, quiet_termi=True)

    return opt2

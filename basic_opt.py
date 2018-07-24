# coding=utf-8
import argparse


def get_basic_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root',
                        type=str,
                        default='output')

    parser.add_argument('-dataset',
                        type=str,
                        default='mini-imagenet')

    # output folder: output/METHOD/DATASET/exp_name
    parser.add_argument('-exp_name', type=str, default='')

    # TRAINING STATS
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-batch_sz',
                        type=int,
                        default=2)

    # MISC
    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('-gpu_id',
                        type=int,
                        nargs='+',
                        help='put -1 if in CPU mode',
                        default=0)
    return parser


def merge_and_setup(opt1, opt2):

    return
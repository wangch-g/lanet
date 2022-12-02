import os
import cv2
import argparse
import numpy as np
import torch
import torchvision

from torchvision import datasets, transforms
from torch.autograd import Variable
from network_v0.model import PointModel
from datasets.hp_loader import PatchesDataset
from torch.utils.data import DataLoader
from evaluation.evaluate import evaluate_keypoint_net


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--device', default=0, type=int, help='which gpu to run on.')
    parser.add_argument('--test_dir', required=True, type=str, help='Test data path.')
    opt = parser.parse_args()

    torch.manual_seed(0)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(opt.device)

    # Load data in 320x240
    hp_dataset_320x240 = PatchesDataset(root_dir=opt.test_dir, use_color=True, output_shape=(320, 240), type='all')
    data_loader_320x240 = DataLoader(hp_dataset_320x240,
                             batch_size=1,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=4,
                             worker_init_fn=None,
                             sampler=None)

    # Load data in 640x480
    hp_dataset_640x480 = PatchesDataset(root_dir=opt.test_dir, use_color=True, output_shape=(640, 480), type='all')
    data_loader_640x480 = DataLoader(hp_dataset_640x480,
                             batch_size=1,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=4,
                             worker_init_fn=None,
                             sampler=None)

    # Load model
    model = PointModel(is_test=True)
    ckpt = torch.load('./checkpoints/PointModel_v0.pth')
    model.load_state_dict(ckpt['model_state'])
    model = model.eval()
    if use_gpu:
        model = model.cuda()


    print('Evaluating in 320x240, 300 points')
    rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(
        data_loader_320x240,
        model,
        output_shape=(320, 240),
        top_k=300)

    print('Repeatability: {0:.3f}'.format(rep))
    print('Localization Error: {0:.3f}'.format(loc))
    print('H-1 Accuracy: {:.3f}'.format(c1))
    print('H-3 Accuracy: {:.3f}'.format(c3))
    print('H-5 Accuracy: {:.3f}'.format(c5))
    print('Matching Score: {:.3f}'.format(mscore))
    print('\n')

    print('Evaluating in 640x480, 1000 points')
    rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net(
        data_loader_640x480,
        model,
        output_shape=(640, 480),
        top_k=1000)

    print('Repeatability: {0:.3f}'.format(rep))
    print('Localization Error: {0:.3f}'.format(loc))
    print('H-1 Accuracy: {:.3f}'.format(c1))
    print('H-3 Accuracy: {:.3f}'.format(c3))
    print('H-5 Accuracy: {:.3f}'.format(c5))
    print('Matching Score: {:.3f}'.format(mscore))
    print('\n')

if __name__ == '__main__':
    main()

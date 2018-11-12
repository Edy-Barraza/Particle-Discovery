import argparse
import os
import sys

import torch
from torchvision import datasets, transforms
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


import numpy as np


def main():
    #people chosen parameters upon running file
    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH',type=str,required=True, help="(str) path to folder with data organized into training, validation, testing")
    parser.add_argument('--shape',type=int,required=True,help="(int) height in pixels for resizing square image")
    parser.add_argument('--bs',type=int,default=1,help="(int, default:1) batch size for these computations")
    parser.add_argument('--max_samples',type=int,default=sys.maxsize,help="(int, default:inf) max number of samples to consider when computing mean & stdev")
    args = parser.parse_args()
    """
    Args:
        PATH (str) - path to folder with data organized into training, validation, testing
        shape (int) - height in pixels for resizing square image
        bs (int, default:1) - batch size for these computations
        max_samples (int, default:inf) - max number of samples to consider when computing mean & stdev

    """

    data_transformations = transforms.Compose([
        transforms.Resize(args.shape),
        transforms.ToTensor()])

    image_datasets = datasets.ImageFolder(os.path.join(args.PATH, 'train'),data_transformations)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=args.bs, shuffle=True)

    pixels = args.shape**2
    size = 0
    mean_sums = [0,0,0]
    for images,_ in dataloaders:
        if size<args.max_samples:
            size+=len(images)
            for i in range(3):
                mean_sums[i] = mean_sums[i] + images.numpy()[:,i,:,:].sum()/(pixels)
        else:
            break
    means = [x/size for x in mean_sums ]

    size =0
    sum_std = [0,0,0]
    for images,_ in dataloaders:
        if size<args.max_samples:
            size+=len(images)
            for i in range(3):
                sum_std[i]=sum_std[i]+((images.numpy()[:,i,:,:] -means[i] )**2).sum()/(pixels)
        else:
            break
    stdevs = [np.sqrt(x/size) for x in sum_std]


    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))


if __name__=='__main__':
    main()

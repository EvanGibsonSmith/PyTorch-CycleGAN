#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from image_models import Generator
from datasets import ImageDataset, SpectrogramDataset

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--height', type=int, default=256, help='height of data')
    parser.add_argument('--width', type=int, default=256, help='width of data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--output_dir', type=str, default="output", help='output directoy')
    parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.width, opt.height)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.width, opt.height)

    # Dataset loader
    #transforms_ = [ transforms.ToTensor(),
    #                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    #dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
    #                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    dataloader = DataLoader(SpectrogramDataset(opt.dataroot, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    for out_dir in ["realA", "realB", "fakeA", "fakeB"]:
        if not os.path.exists(f'{opt.output_dir}/{out_dir}'):
            os.makedirs(f'{opt.output_dir}/{out_dir}')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
        fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

        # Save image files
        torch.save(real_A, f'{opt.output_dir}/realA/%04d.pt' % (i+1))
        torch.save(real_B, f'{opt.output_dir}/realB/%04d.pt' % (i+1))

        torch.save(fake_A, f'{opt.output_dir}/fakeA/%04d.pt' % (i+1))
        torch.save(fake_B, f'{opt.output_dir}/fakeB/%04d.pt' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')
    ###################################

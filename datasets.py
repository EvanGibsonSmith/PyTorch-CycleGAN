import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        # NOTE: Added .convert('RGB') for horse to zebra dataset in which some images only had a single channel
        # Open image and ensure it's in RGB format
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class SpectrogramDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train'):
        if transforms_!=None:
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = None

        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        # NOTE: weights_only argument not relevant for loading mel specs, but supresses annoying warning in terminal 
        # Load mel spectrogram tensors from .pt files
        item_A = torch.load(self.files_A[index % len(self.files_A)], weights_only=False) 

        if self.unaligned: # If no correspondence between spectrograms (which we don't have typically)
            # If unaligned, pick a random sample from B
            item_B = torch.load(self.files_B[random.randint(0, len(self.files_B) - 1)], weights_only=False)
        else:
            item_B = torch.load(self.files_B[index % len(self.files_B)], weights_only=False)  

        # Apply transformations if provided
        if self.transform!=None:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)

        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
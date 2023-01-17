from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import scipy.io
import torchvision.transforms as tvtf



class Samson(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        """Init Samson dataset."""
        super(Samson, self).__init__()

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform


        PATH = "HyperspecAE/data/data.mat"

        training_data = scipy.io.loadmat(PATH)

        self.train_data = training_data['V'].T

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is abundance fractions for each pixel.
        """
        
        img, target = self.train_data[index], self.labels[index]
        

        if self.transform is not None:
            img = torch.tensor(img)

        if self.target_transform is not None:
            target = torch.tensor(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""

        return len(self.train_data)
        
        
def get_dataloader(BATCH_SIZE: int, DIR):
    """Create a DataLoader for input data stream."""
    trans = tvtf.Compose([tvtf.ToTensor()])

    # Load train data
    source_domain = Samson(root=DIR, transform=trans, target_transform=trans)
    source_dataloader = torch.utils.data.DataLoader(source_domain, BATCH_SIZE)
    
    return source_dataloader, source_domain
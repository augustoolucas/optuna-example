import torch
from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=0,
                      pin_memory=True,
                      shuffle=True)

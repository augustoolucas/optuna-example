import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, shuffle=True
)

class Digits(Dataset):
    def __init__(self, train=True):
        super().__init__()

        self.data = X_train if train else X_test
        self.targets = y_train if train else y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float()
        target = self.targets[idx]
       
        return img, target

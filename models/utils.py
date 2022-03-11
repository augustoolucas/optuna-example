import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score#, precision_score, confusion_matrix, ConfusionMatrixDisplay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device == torch.device('cuda')

class Trainer():
    def __init__(self, model, dataloader, optim, lr, wd=None):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.loss_fn = nn.CrossEntropyLoss()
        
        if optim == 'Adam':
            self.optim = torch.optim.Adam(params=self.model.parameters(),
                                          lr=lr,
                                          weight_decay = wd if wd else 0)
        elif optim == 'SGD':
            self.optim = torch.optim.SGD(params=self.model.parameters(),
                                         lr=lr,
                                         weight_decay = wd if wd else 0)

    def train(self, epochs):
        self.model.train()

        train_bar = tqdm(range(epochs))
        for epoch in train_bar:
            epoch_loss, epoch_acc = 0, 0
            for imgs, targets in self.dataloader:
                self.model.zero_grad()
                imgs, targets = imgs.to(device), targets.to(device)
                model_output = self.model(imgs)

                loss = self.loss_fn(model_output, targets)
                loss.backward()
                self.optim.step()

                ### ------ Training Metrics ------ ###
                epoch_loss += loss.item()
                predictions = torch.argmax(model_output, dim=1).tolist()
                epoch_acc += accuracy_score(predictions, targets.detach().cpu().numpy())

            train_bar.set_description(f'Training Loss: {(epoch_loss/len(self.dataloader)):.03f} - Accuracy: {(epoch_acc/len(self.dataloader)):.03f}') 
        return epoch_loss/len(self.dataloader)


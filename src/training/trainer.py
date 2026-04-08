import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        losses = []

        for img, mask in tqdm(loader):
            img, mask = img.to(self.device), mask.to(self.device)

            pred = self.model(img)
            loss = self.criterion(pred, mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return sum(losses) / len(losses)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from copy import deepcopy
import math
import os
import sys

sys.path.append(os.path.abspath("."))

from backbone import LeNet
from utils.data import load_data


class SSL_BYOL(nn.Module):
    # Based on Bootstrap Your Own Latent Architecture

    def __init__(self, num_classes, device="cpu"):
        super().__init__()

        self.encoder = LeNet(num_classes)
        self.encoder.fc3 = nn.Identity()  # don't want predictions, just representation

        self.projector = nn.Sequential(
            nn.Linear(84, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32)
        )

        self.predictor = nn.Sequential(
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32)
        )

        self.online_network = nn.Sequential(
            self.encoder,
            self.projector,
            self.predictor
        )

        self.target_network = deepcopy(nn.Sequential(
            self.encoder,
            self.projector
        ))

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.online_network.parameters())  # only want to optimize online network
        self.tau_base = 0.996  # hyperparameter
        self.tau = self.tau_base  # to be modified later

        self.device = device

        self.online_augmentation = transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop((28, 28))], p=0.7),

            transforms.RandomHorizontalFlip(),

            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),

            transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.2),

            transforms.GaussianBlur((3, 3))
        ])
        
        self.target_augmentation = transforms.Compose([

            transforms.RandomApply([transforms.CenterCrop((28, 28))], p=0.3),
            transforms.RandomHorizontalFlip(),

            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),

            transforms.RandomApply([transforms.GaussianBlur((3, 3))], p=0.1),

            transforms.RandomSolarize(threshold=0.5, p=0.2)
        ])

    def fit(self, train_loader, val_loader, epochs=1000, print_every=100):

        train_loss = []
        val_loss = []

        model_save_path = os.path.join("checkpoints", "ssl_byol")
        encoder_save_path = os.path.join("checkpoints", "ssl_byol_encoder")

        for epoch in range(epochs):
            
            self.train(train_loader)

            # self.tau += 1 - (1 - self.tau_base) * (math.cos(math.pi*(epoch + 1) / epochs) + 1)/2

            train_loss.append(self.validate(train_loader))
            val_loss.append(self.validate(val_loader))
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch}")
                print(f"Train Loss: {train_loss}")
                print(f"Validation Loss: {val_loss}")

                # Full model:
                torch.save(self.state_dict(), os.path.join(model_save_path, f"checkpoint_{epoch + 1}.pt"))
                # Just encoder:
                torch.save(self.encoder.state_dict(), os.path.join(encoder_save_path, f"checkpoint_{epoch + 1}.pt"))
        
        return train_loss, val_loss

    def train(self, train_loader):

        self.online_network.train()
        self.target_network.train()

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()

            batch_loss = self.forward(batch_X)

            batch_loss.backward()
            self.optimizer.step()

            # Update target's parameters
            for online_param, target_param in zip(self.online_network.parameters(), self.target_network.parameters()):
                target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data

        self.online_network.eval()
        self.target_network.eval()
    
    def forward(self, batch):
        view = self.online_augmentation(batch)
        view_prime = self.target_augmentation(batch)

        online_output, online_output_prime = self.online_network(view), self.online_network(view_prime)
        online_output, online_output_prime = F.normalize(online_output), F.normalize(online_output_prime)

        with torch.no_grad():
            target_output, target_output_prime = self.target_network(view_prime), self.target_network(view)
            target_output, target_output_prime = F.normalize(target_output), F.normalize(target_output_prime)

        # symmetric loss
        loss = self.loss_fn(online_output, target_output) + self.loss_fn(online_output_prime, target_output_prime)
        return loss

    def validate(self, loader):
        loss = 0
        count = 0

        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            with torch.no_grad():
                loss += self.forward(batch_X)
            count += len(batch_X)
        
        return loss / count


def train_byol(): 
    data = load_data()
    
    train_data = torch.utils.data.ConcatDataset([data["id_train_seen"],
                                                 data["ood_train_seen"]])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data["id_val_seen"])

    ssl_byol = SSL_BYOL(num_classes=5)

    ssl_byol.fit(train_loader, val_loader, epochs=20, print_every=5)


def forward_dataset(encoder, dataset):
    encoder.eval()
    embeddings = None
    labels = []

    for X, y in dataset:
        embedding = encoder(X[None])

        labels.append(y)

        if embeddings is None:
            embeddings = embedding
        else:
            embeddings = torch.concat((embeddings, embedding), dim=0)

    encoder.train()
    return embeddings, labels


def test_byol():
    data = load_data()
    id_test = data["id_test_seen"]
    ood_test = data["ood_test_unseen"]

    encoder = LeNet()
    encoder.fc3 = nn.Identity()
    encoder_path = os.path.join("checkpoints", "ssl_byol_encoder", "checkpoint_10.pt")
    encoder.load_state_dict(torch.load(encoder_path))


if __name__=="__main__":
    test_byol()

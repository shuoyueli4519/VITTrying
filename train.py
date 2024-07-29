import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from model import Vit

parser = argparse.ArgumentParser(description='Pytorch ViT training based on cifar10')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adam")
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='10')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)

args = parser.parse_args()

bs = int(args.bs)
imsize = int(args.size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"==> Preparing data...")
size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(size, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(f"==> Building model...")

net = Vit(image_size=size, patch_size=int(args.patch), num_classes=10, dim=512, depth=12, heads=8, mlp_dim=512, dim_head=int(args.dimhead), dropout=0.1, emb_dropout=0.1).to(device)
criterion = nn.CrossEntropyLoss()
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.n_epochs))

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

if __name__ == '__main__':
    for epoch in range(int(args.n_epochs)):
        train_loss = train(epoch)
        scheduler.step()
        print(f"Epoch {epoch} finished, loss: {train_loss}")
    net = net.to('cpu')
    torch.save(net.state_dict(), f"vit.pkl")
    

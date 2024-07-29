import torchvision
import torchvision.transforms as transforms
import torch
import argparse

from model import Vit

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
print(f"==> Preparing data...")
size = imsize

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = Vit(image_size=size, patch_size=int(args.patch), num_classes=10, dim=512, depth=12, heads=8, mlp_dim=512, dim_head=int(args.dimhead), dropout=0.1, emb_dropout=0.1).to(device)
net.load_state_dict(torch.load('vit.pkl', weights_only=True))
net.eval()

test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        print('Acc: %.3f%% (%d/%d)' %(100.*correct/total, correct, total))
import argparse
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import BayesNet


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 784)

        model.zero_grad()
        outputs = model.sample_elbo(
            data, target, len(train_loader), args.num_samples)

        loss = outputs[0]
        nll = outputs[1]
        kl = outputs[2]
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tNLL: {:.6f}, KL: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), nll.item(), kl.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)
            logprobs = 0
            for _ in range(args.num_samples):
                logprob = model(data)
                logprobs += logprob
            # sum up batch loss
            logprob = logprobs / args.num_samples
            test_loss += F.nll_loss(logprob ,
                                    target, reduction='sum').item()
            
            # get the index of the max log-probability
            pred = logprob.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def run(args):
    device = torch.cuda.current_device()
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            './fmnist', train=True, download=True,
            transform=transform),
        batch_size=args.batch_size, shuffle=True,)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            './fmnist', train=False, download=True,
            transform=transform),
        batch_size=args.batch_size, shuffle=False, )

    model = BayesNet(args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument("--input_size", type=int, default=784)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--output_size", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--pi", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=5)

    args = parser.parse_args()
    args.sigma_1 = math.exp(0)
    args.sigma_2 = math.exp(-6)
    run(args)

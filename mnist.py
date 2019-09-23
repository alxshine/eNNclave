""" Test script for a pytorch MNIST CNN"""

import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import interop.pymatutil as pymatutil


class Net(nn.Module):
    """ Defines the neural net for testing """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=True)
        self.fc2 = nn.Linear(500, 10, bias=True)

    def dense(self, batch):
        state = self.state_dict()
        m1 = state['fc1.weight'].detach().numpy().astype(np.float32).T
        r1 = m1.shape[0]
        c1 = m1.shape[1]
        mb1 = m1.tobytes()
        bb1 = state['fc1.bias'].detach().numpy().tobytes()

        m2 = state['fc2.weight'].detach().numpy().astype(np.float32).T
        r2 = m2.shape[0]
        c2 = m2.shape[1]
        mb2 = m2.tobytes()
        bb2 = state['fc2.bias'].detach().numpy().tobytes()

        output = np.zeros((batch.shape[0], self.fc2.out_features))

        for i, x in enumerate(batch):
            x = x.detach().numpy().astype(np.float32)
            xr = 1
            xc = x.shape[0]
            xb = x.tobytes()

            mul1 = pymatutil.multiply(xb, xr, xc, mb1, r1, c1)
            add1 = pymatutil.add(mul1, xr, c1, bb1, xr, c1)
            fc1_out = pymatutil.relu(add1, xr, c1)

            mul2 = pymatutil.multiply(fc1_out, xr, c1, mb2, r2, c2)
            add2 = pymatutil.add(mul2, xr, c2, bb2, xr, c2)
            output[i] = np.frombuffer(add2, dtype=np.float32)

        return torch.Tensor(output)

    def full_dense(self, batch):
        labels = np.empty((batch.shape[0]))
        for i, x in enumerate(batch):
            x = x.detach().numpy().astype(np.float32)
            xr = 1
            xc = x.shape[0]
            xb = x.tobytes()

            labels[i] = pymatutil.dense(xb, xr, xc)
        return labels

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        x_test = F.relu(self.fc1(x))
        x_test = self.fc2(x_test)
        ret_test = F.log_softmax(x_test, dim=1)

        # x_self = self.dense(x)  # move to C
        # ret_self = F.log_softmax(x_self, dim=1)
        # breakpoint()

        labels = self.full_dense(x)

        if np.array_equal(ret_test.argmax(
                dim=1), labels):  # check my dense code for correct labels
            print("Labels are identical with reference torch output")
        else:
            print(
                "ERROR: Labels are NOT identical with reference torch output")

        return labels

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            correct += (output == target.numpy()).sum()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        'data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum)

    save_file = "models/mnist_cnn.pt"
    if os.path.isfile(save_file):
        print('loading from %s' % save_file)
        model.load_state_dict(torch.load(save_file), strict=False)
    else:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

        # if args.save_model:
        torch.save(model.state_dict(), save_file)

    # run some eval code
    pymatutil.initialize()
    test(args, model, device, test_loader)
    pymatutil.teardown()

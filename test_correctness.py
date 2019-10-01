""" Test script for a pytorch MNIST CNN"""

import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import interop.pymatutil as pymatutil


class Net(nn.Module):
    """ Defines the neural net for testing """
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    # def dense(self, batch):
    #     state = self.state_dict()
    #     m1 = state['fc1.weight'].detach().numpy().astype(
    #         np.float32).T
    #     r1 = m1.shape[0]
    #     c1 = m1.shape[1]
    #     mb1 = m1.tobytes()
    #     bb1 = state['fc1.bias'].detach().numpy().tobytes()

    #     m2 = state['fc2.weight'].detach().numpy().astype(
    #         np.float32).T
    #     r2 = m2.shape[0]
    #     c2 = m2.shape[1]
    #     mb2 = m2.tobytes()
    #     bb2 = state['fc2.bias'].detach().numpy().tobytes()

    #     output = np.zeros((batch.shape[0], self.fc2.out_features))

    # for i, x in enumerate(batch):
    #     x = x.detach().numpy().astype(np.float32)
    #     xr = 1
    #     xc = x.shape[0]
    #     xb = x.tobytes()

    #     mul1 = pymatutil.multiply(xb, xr, xc, mb1, r1, c1)
    #     add1 = pymatutil.add(mul1, xr, c1, bb1, xr, c1)
    #     fc1_out = pymatutil.relu(add1, xr, c1)

    #     mul2 = pymatutil.multiply(fc1_out, xr, c1, mb2, r2, c2)
    #     add2 = pymatutil.add(mul2, xr, c2, bb2, xr, c2)
    #     output[i] = np.frombuffer(add2, dtype=np.float32)

    # return torch.Tensor(output)

    def forward(self, x):
        return self.fc(x)


def np_to_bytes(x):
    return x.astype(np.float32).tobytes()


def multiply(m1, m2):
    """ takes two numpy matrices and multiplies them using the C implementation
    RETURNS: numpy matrix containing the matrix multiplication result """
    r1, c1 = m1.shape
    r2, c2 = m2.shape
    b1 = m1.astype(np.float32).tobytes()
    b2 = m2.astype(np.float32).tobytes()
    bret = pymatutil.multiply(b1, r1, c1, b2, r2, c2)
    return np.frombuffer(bret, dtype=np.float32)


if __name__ == '__main__':
    IN_FEATURES = 4 * 4 * 50
    OUT_FEATURES = 500

    for i in range(10):
        fc = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        w = fc.weight.detach().numpy().T
        x = np.random.normal(size=((1, IN_FEATURES)))

        r_torch = fc(torch.Tensor(x)).detach().numpy()
        r_numpy = np.matmul(x, w)
        r_c = multiply(x, w)
        d_tn = r_torch - r_numpy
        d_tc = r_torch - r_c
        d_nc = r_numpy - r_c

        print("Run %d" %(i+1))
        print("d_tn:")
        print("Mean: %f, Max: %f, Min: %f" % (np.mean(d_tn), np.max(d_tn),
              np.min(d_tn)))

        print("d_tc:")
        print("Mean: %f, Max: %f, Min: %f" % (np.mean(d_tc), np.max(d_tc),
              np.min(d_tc)))

        print("d_nc:")
        print("Mean: %f, Max: %f, Min: %f" % (np.mean(d_nc), np.max(d_nc),
              np.min(d_nc)))

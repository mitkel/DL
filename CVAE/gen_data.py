import numpy as np
import torch
from torch.utils.data import Dataset

def EM_step(x,dt=1,mu=0,sigma=1,theta=1, size=None, **kwargs):
    x = np.array(x)
    Z = np.random.normal(size=size)
    return x + theta*(mu-x)*dt + np.sqrt(2*dt)*sigma*Z

def stationary(size=None, mu=0., sigma=1., theta=1., **kwargs):
    return np.random.normal(loc = mu, scale = sigma/np.sqrt(theta), size=size)

# slices - # slices of the interval [0,1]
def trajectory(x0, T, slices=1, **kwargs):
    X=np.array(x0)
    x,y=x0,x0
    for i in range(T*slices):
        x, y = y, EM_step(x, dt=1/slices, **kwargs)
        if i%slices == slices-1:
            X = np.append(X, y)
    return X

class cvae_ds(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype = torch.float)
        self.Y = torch.tensor(Y, dtype = torch.float)

        if len(self.X.shape) == 1:
            self.X = self.X.unsqueeze(1)

        if len(self.Y.shape) == 1:
            self.Y = self.Y.unsqueeze(1)


    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if self.X.shape == (1,):
            return (self.X, self.Y[idx])
        else:
            return (self.X[idx], self.Y[idx])

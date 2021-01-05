import torch
import torch.optim as optim

params = {
    # data
    'mu': 1.,
    'sigma': 0.2,
    'theta': 1.,
    'T': 330,
    'M': 30,

    # samples
    'train_samples': 10000,
    'test_samples': 1000,
    'slices': 10,

    # model
    'latent_dim': 50,
    'bias': True,
    'layers': [1000,500,100],

    # training
    'epochs': 10,
    'batch_size': 100,
    "optimizer": optim.Adam,
    "lr": 1e-3,
}

params.update({
    'input_dim': params['M'],
    'cond_dim': params['T']-params['M']+1,
    'name': f"res/CVAE-L{params['layers']}-C{params['latent_dim']}",
    'reg': torch.ones(1),
})

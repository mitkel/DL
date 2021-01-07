import torch
import torch.optim as optim

params = {
    # data
    'mu': 1.,
    'sigma': 0.2,
    'theta': 1.,
    'input_dim': 100,
    'cond_dim': 10,

    # samples
    'train_samples': 10000,
    'slices': 10,

    # model
    'latent_dim': 50,
    'bias': False,
    'layers': [1000,500,100],
    'end_relu': False,

    # training
    'epochs': 100,
    'batch_size': 100,
    "optimizer": optim.Adam,
    "lr": 1e-3,
}

params.update({
    'name': f"res/CVAE-L{params['layers']}-C{params['latent_dim']}",
    'reg': torch.ones(1),
})

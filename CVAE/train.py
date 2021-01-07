import torch
import numpy as np

def train(model, train_loader, device, verbose=False, **kwargs):
    optimizer = kwargs['optimizer'](model.parameters(), lr=kwargs['lr'])
    model.train()

    for epoch in range(kwargs['epochs']):
        loss = torch.zeros((), device = device)
        MSE = torch.zeros((), device = device)
        KLD = torch.zeros((), device = device)

        for cond, inpt in train_loader:
            cond = cond.to(device)
            inpt = inpt.to(device)
            optimizer.zero_grad()

            outputs, mu, logvar = model(inpt, cond)
            train_loss, MSE_train, KLD_train = model.loss_func(outputs, inpt, mu, logvar)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
            MSE += MSE_train.item()
            KLD += KLD_train.item()

        loss = loss / len(train_loader)
        MSE = MSE / len(train_loader)
        KLD = KLD / len(train_loader)
        if verbose:
            print("{}: L {:.5f} MSE {:.5f} KLD {:.5f}".format(epoch + 1,
                                                              loss.item(),
                                                              MSE.item(),
                                                              KLD.item()))

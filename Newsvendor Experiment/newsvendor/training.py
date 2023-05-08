import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def train_model(model, 
                loss, 
                dloader,
                n_epochs=10000, 
                steplr_size=1000, 
                steplr_gamma=0.5, 
                plot_convergence=False, 
                usetqdm=True):
    '''
    Train Model

    Args:
        model               : model to train
        loss                : Loss (from loss_functions)
        dloader             : Dataloader to use
        n_epochs            : Number of epochs to train
        steplr_size         : stepsize of the StepLR
        steplr_gamma        : gamma of the StepLR
        plot_convergence    : plot convergence plot at the end of training?
        usetqdm             : use tqdm to visualize training progress

    Returns:
        model               : the trained model (the same as argument model)

    '''
    # train the model
    losses = []
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=steplr_size, gamma=steplr_gamma)
    for e in tqdm(range(n_epochs), disable=not usetqdm):
        xis, ps = dloader()
        optimizer.zero_grad()
        x_hat = model(xis)
        loss_value = loss(x_hat, (xis, ps)).mean()
        loss_value.backward()
        optimizer.step()
        losses.append(loss_value.item())
        scheduler.step()

    # plot convergence
    if plot_convergence:
        f, (ax1) = plt.subplots(1, 1)
        ax1.plot(losses)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
    
    return model

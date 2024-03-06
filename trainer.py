# ML specific Modules
import torch
from torch import nn
from torch.optim import Adam

# Side Modules
from time import time
from importlib import reload

# My Modules
import dataset
reload(dataset)

def train_one_epoch(model, lossfn, optimizer, trainloader, debug=False):
    if debug:
        print("Training one epoch")
    
    device = next(model.parameters()).device

    i = 0
    train_loss = 0
    train_size = 0
    start = time()
    model.train()


    for x in trainloader:
        x = x.to(device)

        # Prediction Part
        reconstruction = model(x)
        loss = lossfn(reconstruction, x)

        # Optimization Part
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss data Part
        train_loss += loss.cpu().detach().numpy()
        train_size += x.shape[0].cpu().detach().numpy()

        if debug:
            i += 1
            print(f"Trained on batch {i}")
    
    end = time()
    time = end - start
    loss = train_loss/train_size

    if debug:
        print(f"Time taken = {time} s, and train loss = {loss}")

    return time, loss

def validate(model, lossfn, validation_loader, debug=False):
    with torch.no_grad():
        device = next(model.parameters).device
        if debug:
            print("Validating one epoch")
        
        i = 0
        validation_loss = 0
        validation_size = 0
        start = time()
        for x in validation_loader:
            x = x.to(device)
            reconstruction = model(x)
            loss = lossfn(reconstruction, x)
            validation_loss += loss.cpu().detach().numpy()
            validation_size += x.shape[0].cpu().detach().numpy()

            if debug:
                i += 1
                print(f"Validation on the batch {i}")
        
        end = time()
        time = end - start
        loss = validation_loss/validation_size

        if debug:
            print(f"Time taken = {time} s, and validation loss = {loss}")
        
        return time, loss


    
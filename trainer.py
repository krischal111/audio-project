# ML specific Modules
import torch
from torch import nn
from torch.optim import Adam

# Side Modules
from time import time
from importlib import reload
from pathlib import Path

# My Modules
# import dataset
# reload(dataset)

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
        # loss = torch.sum(reconstruction,dim=-1)

        # Optimization Part
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss data Part
        train_loss += loss.cpu().detach().numpy()
        train_size += x.shape[0]

        if debug:
            i += 1
            print(f"Trained on batch {i}")
    
    end = time()
    totaltime = end - start
    loss = train_loss/train_size

    if debug:
        print(f"Time taken = {time} s, and train loss = {loss}")

    return totaltime, loss

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
            validation_size += x.shape[0]

            if debug:
                i += 1
                print(f"Validation on the batch {i}")
        
        end = time()
        time = end - start
        loss = validation_loss/validation_size

        if debug:
            print(f"Time taken = {time} s, and validation loss = {loss}")
        
        return time, loss

def save_model(model : nn.Module, name : str, statistics : dict = None):
    ''' 
    model: The model to be saved.
    statistics: The dictionary type with series of train losses and validation losses.
    name: Name of the model to be saved.

    Effect 1: Model's state dict will be saved at the f"models/{name}/model_state_dict.pth"
    Effect 2: Statistics would be saved to f"models/{name}/statistics.pth"
    Return value: None
    '''

    model_directory = Path("models") / Path(name)
    model_directory.mkdir(parents=True, exist_ok=True)
    model_file = model_directory / Path("model_state_dict.pth")
    statistics_file = model_directory / Path("statistics.pth")

    torch.save(model.state_dict(), model_file)
    print(f"Saved model to {model_file}")
    if statistics is not None:
        torch.save(statistics, statistics_file)
        print(f"Saved statistics to {statistics_file}")

def retrieve_model(model, model_path):
    ''' 
    model: The model whose weights are to be loaded.
    model_path: The path where the model is located (its weights and stats)

    Effect: model will have its weights initialized
    Return value: statistics file
    '''
    model_path = Path(model_path)
    model_file = model_path / Path("model_state_dict.pth")
    statistics_file = model_path / Path("statistics.pth")

    stats = torch.load(statistics_file)
    model.load_state_dict(torch.load(model_file))

    return stats
    

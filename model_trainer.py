# ML specific
from torch import nn
import torch

# Extras
from time import time
from pathlib import Path

def train_one_batch(model, lossfn, optimizer, x, debug=False):
    # setting up model and data
    model.train()
    device = next(model.parameters()).device
    x = x.to(device)
    starttime = time()

    if debug:
        print("Training step:")
        print("Forward propagating... ", end=None)
    pred, qloss = model.train_forward(x)
    loss = lossfn(pred, x) + qloss

    forwarddonetime = time()
    forward_time = forwarddonetime - starttime
    if debug:
        print(f"Done. Time taken = {forward_time}")


    if debug:
        print("Back Propagating... ", end=None)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    backpropdonetime = time()
    backprop_time = backpropdonetime - forwarddonetime
    if debug:
        print(f"Done. Time taken = {backprop_time}")

    # statistics
    loss = loss.cpu().detach().numpy()
    if debug:
        print(f"Saving statistics: Loss (avg) = {loss}\n")

    return loss, forward_time, backprop_time



def validate_one_batch(model, lossfn, x, debug=False):
    # setting up model and data
    with torch.no_grad():
        device = next(model.parameters()).device
        x = x.to(device)
        starttime = time()

        if debug:
            print("Validation step:")
            print("Forward propagating... ", end=None)
        pred, qloss = model.train_forward(x)
        loss = lossfn(pred, x) + qloss

        forwarddonetime = time()
        forward_time = forwarddonetime - starttime
        if debug:
            print(f"Done. Time taken = {forward_time}")

        # statistics
        loss = loss.cpu().detach().numpy()
        if debug:
            print(f"Saving statistics: Loss (avg) = {loss}\n")

        return loss, forward_time



def train_and_validate(model:nn.Module, lossfn, optimizer, trainloader, validationloader, debug=False, limit=0):
    if debug:
        if not limit:
            print("Training for an epoch")
        else:
            print(f"Training for {limit} batches.")
    
    train_losses = []
    validation_losses = []
    total_train_loss = 0
    total_validation_loss = 0
    train_forward_times = []
    back_prop_times = []
    validation_forward_times = []
    i = 0

    for (trainx, validationx) in zip(trainloader, validationloader):
        i += 1
        if limit and i > limit:
            break

        if debug:
            print(f"\nBatch {i}:")
            print("==========")
        
        train_loss, train_forward_time, back_prop_time = train_one_batch(model, lossfn, optimizer, trainx, debug)
        train_losses.append(train_loss)
        total_train_loss += train_loss

        validation_loss, validation_forward_time = validate_one_batch(model, lossfn, validationx, debug)
        validation_losses.append(validation_loss)
        total_validation_loss += validation_loss

        train_forward_times.append(train_forward_time)
        back_prop_times.append(back_prop_time)
        validation_forward_times.append(validation_forward_time)
    
    statistics = {
        "TrainLosses":train_losses,
        "ValidationLosses":validation_losses,
        "TotalTrainLoss":total_train_loss,
        "TotalValidationLoss":total_validation_loss,
        "TrainForwardTimes":train_forward_times,
        "BackPropTimes":back_prop_times,
        "ValidationForwardTimes":validation_forward_times,
        "NBatches": i,
    }
    return statistics

def null_stats():
    statistics = {
        "TrainLosses":[],
        "ValidationLosses":[],
        "TotalTrainLoss":0,
        "TotalValidationLoss":0,
        "TrainForwardTimes":[],
        "BackPropTimes":[],
        "ValidationForwardTimes":[],
        "NBatches": 0,
    }
    return statistics

def concatenate_stats(stat1, stat2):
    l1 = stat1["NBatches"]
    l2 = stat2["NBatches"]
    n = l1 + l2
    l1 /= n
    l2 /= n
    concatenated_statistics = {
        "TrainLosses": [*stat1["TrainLosses"], *stat2["TrainLosses"],],
        "ValidationLosses":[*stat1["ValidationLosses"], *stat2["ValidationLosses"],],
        "TrainForwardTimes":stat1["TrainForwardTimes"] + stat2["TrainForwardTimes"],
        "BackPropTimes":stat1["BackPropTimes"] + stat2["BackPropTimes"],
        "ValidationForwardTimes":stat1["ValidationForwardTimes"] + stat2["ValidationForwardTimes"],

        "TotalTrainLoss":stat1["TotalTrainLoss"] *l1 + stat2["TotalTrainLoss"] * l2,
        "TotalValidationLoss":stat1["TotalValidationLoss"] + stat2["TotalValidationLoss"],
        "NBatches": n,
    }
    return concatenated_statistics
    

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

    if model_file.exists():
        model.load_state_dict(torch.load(model_file))
        print(f"Model loaded from {model_file}")
    else:
        raise FileNotFoundError("Model file doesn't exist!")

    if statistics_file.exists():
        stats = torch.load(statistics_file)
        print(f"Statistics loaded from {statistics_file}")
        return stats
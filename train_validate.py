from torch import nn
from time import time
import torch

class timeclass:
    def __init__(self):
        pass

    def start(self, val):
        self.starttime = val 

    def prediction(self, val):
        self.predictiontime = val 

    def backprop(self, val):
        self.backproptime = val 

    def validation(self, val):
        self.validationtime = val 

    def get_3_times(self):
        return (self.predictiontime - self.starttime, 
                self.backproptime - self.predictiontime,
                self.validationtime-self.backproptime)
    
    def __repr__(self):
        pred, bprop, valid = self.get_3_times()
        return f'''
        Time for prediction: {pred}
        Time for backpropagation: {bprop}
        Time for validation: {valid}
    '''

def train_and_validate_simul(model:nn.Module, lossfn, optimizer, trainloader, validationloader, debug=False):
    if debug:
        print("Training for an epoch")
    
    device = next(model.parameters()).device

    train_losses = []
    validation_losses = []
    train_size = 0
    validation_size = 0
    

    logtime = timeclass()
    logtime.start(time())
    i = 0

    model.train()
    for (x_t, x_v) in zip(trainloader, validationloader):
        x = x_t.to(device)

        # prediction
        recons = model(x)
        t_loss = lossfn(recons, x)
        logtime.prediction(time())

        if debug:
            i += 1
            print(f"Predicted {i} th train batch: loss = {t_loss}")

        # optimization
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        logtime.backprop(time())

        if debug:
            print(f"Backpropagated for the {i} th batch.")

        # saving loss statistics
        t_loss = t_loss.cpu().detach().numpy()
        train_losses.append(t_loss)
        train_size += x.shape[0]

        with torch.no_grad():
            # prediction and loss
            x = x_v.to(device)
            recons = model(x)
            v_loss = lossfn(recons, x)
            logtime.validation(time())

            if debug:
                print(f"Validation loss for {i} th batch : {v_loss}")

            # loss stats saving
            v_loss = v_loss.cpu().detach().numpy()
            validation_losses.append(v_loss)
            validation_size += x.shape[0]
            pass # validation batch
        pass # train loop
    avg_train_loss = sum(train_losses) / train_size
    avg_valid_loss = sum(validation_losses) / validation_size
    if debug:
        print(logtime)
        print(f"Train Loss = {avg_train_loss}")
        print(f"Validation Loss = {avg_valid_loss}")
    return logtime, avg_train_loss, avg_valid_loss
        
        

            

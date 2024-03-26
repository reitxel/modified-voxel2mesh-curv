import logging
import torch
# from torch.utils.tensorboard import SummaryWriter
from utils.utils_common import DataModes

import numpy as np
from torch.optim import lr_scheduler
import time 
import wandb
from IPython import embed
import time 
logger = logging.getLogger(__name__)



class Trainer(object):
    def __init__(self, net, trainloader, val_loader, optimizer, numb_of_itrs, eval_every, save_path, evaluator):

        self.net = net
        self.trainloader = trainloader
        self.val_loader = val_loader
        self.optimizer = optimizer 

        self.numb_of_itrs = numb_of_itrs
        self.eval_every = eval_every
        self.save_path = save_path 

        self.evaluator = evaluator  
        
        # added this for model checkpoint
        self.best_val_loss = float('inf')  # initialize the best validation loss as infinity
        
        # included a learning rate scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.1, verbose=True)
 
    def training_step(self, data, epoch):
        # Get the minibatch 
        self.optimizer.zero_grad()
        loss, log = self.net.loss(data, epoch) 
        loss.backward()
        self.optimizer.step()  

        return log
    
    def validation_step(self, data, epoch): 
        # perform validation step
        with torch.no_grad():  # disable gradient computation during validation
            val_loss, val_log = self.net.loss(data, epoch)  # compute validation loss
        return val_loss, val_log
    
    def validate(self, epoch):
        # iterate over validation dataset once and return log
        self.net = self.net.eval()  # set the model to evaluation mode
        val_losses = []  # store validation losses to compute average loss later
        
        for data in self.val_loader:  # iterate through the validation dataset
            val_loss, _ = self.validation_step(data, epoch)  # perform a validation step
            val_losses.append(val_loss.item())  # new: store the validation loss
        
        avg_val_loss = np.mean(val_losses)  # compute the average validation loss
        log_vals = {'avg_val_loss': avg_val_loss, 'epoch': epoch}    
        wandb.log(log_vals) # load loss into wandb
        
        self.net = self.net.train()  # revert the model back to training mode
        return avg_val_loss
    
    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss  # save the best validation loss
        }, f"{self.save_path}/best_model.pth")
        
    def train(self, start_iteration=1):

        print("Start training...") 
 
        self.net = self.net.train()
        iteration = start_iteration 
        print_every = 1
        # implementing early stopping, counter and threshold
        patience_counter = 0
        max_patience = 20
        
        # for epoch in range(10000000):  # loop over the dataset multiple times
        for epoch in range(9000):
            print('Epoch number:', epoch)
 
            for itr, data in enumerate(self.trainloader):
  
                # training step 
                loss = self.training_step(data, start_iteration)

                if iteration % print_every == 0:
                    log_vals = {}
                    for key, value in loss.items():
                        log_vals[key] = value / print_every 
                    log_vals['iteration'] = iteration 
                    wandb.log(log_vals)

                iteration = iteration + 1 

                if iteration % self.eval_every == self.eval_every-1:  # print every K epochs
                    val_loss = self.validate(epoch)
                    self.evaluator.evaluate(iteration)
                    
                    # added model checkpoint
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch)
                        
                    # update learning scheduler
                    self.scheduler.step(val_loss)
                   
                # early-stopping check
                if patience_counter >= max_patience:
                    print("Early stopping triggered")
                    break

                if iteration > self.numb_of_itrs:
                    break
   

        logger.info("... end training!")

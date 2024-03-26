 
import os
GPU_index = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
import torch
import numpy as np
from train import Trainer
from evaluate import Evaluator  

from shutil import copytree, ignore_patterns
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_common import DataModes
import wandb
from IPython import embed 
from utils.utils_common import mkdir

from config import load_config
from voxel2mesh import Voxel2Mesh as network

logger = logging.getLogger(__name__)

def main(config=None):
    
    exp_id = 3

    # Initialize
    cfg = load_config(exp_id)
    #trial_path, trial_id = init(cfg) 
    trial_path = cfg.save_path 
    trial_id = 1
    
    # define default random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    
    # for wandb parameter tuning
    # with wandb.init(config=config):
        #sweep_config = wandb.config
 
    print("Create network")
    classifier = network(cfg)
    classifier.cuda()

    print("Initialize wandb")

    wandb.init(name='A1_segment-improvs-cl-test', entity="reitxel", project="voxel2mesh_v5", dir=trial_path)
    # wandb.init(sweep_config, name='A1_segment-baseline-64-seed-sweep', entity="reitxel", project="voxel2mesh_v2", dir=trial_path)
    
    print("Initialize optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=cfg.learning_rate)  

    print("Load pre-processed data") 
    data_obj = cfg.data_obj 
    data = data_obj.quick_load_data(cfg, trial_id)

    loader = DataLoader(data[DataModes.TRAINING], batch_size=classifier.config.batch_size, shuffle=True)
    val_loader = DataLoader(data[DataModes.VALIDATION], batch_size=classifier.config.batch_size, shuffle=True)

    print("Trainset length: {}".format(loader.__len__()))
    print("Valset length: {}".format(val_loader.__len__()))

    print("Initialize evaluator")
    evaluator = Evaluator(classifier, optimizer, data, trial_path, cfg, data_obj) 

    print("Initialize trainer")
    trainer = Trainer(classifier, loader, val_loader, optimizer, cfg.numb_of_itrs, cfg.eval_every, trial_path, evaluator)

    if cfg.trial_id is not None:
        print("Loading pretrained network")
        save_path = trial_path + '/best_performance3/model.pth'
        checkpoint = torch.load(save_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0


    trainer.train(start_iteration=epoch) 

    # To evaluate a pretrained model, uncomment line below and comment the line above
    # evaluator.evaluate(epoch, splitmode=DataModes.TESTING)

if __name__ == "__main__": 
    main()
    
    # hyperparameter tuning of loss terms
#     sweep_config = {
#         "method": "random",
#         "metric": {"goal": "minimize", "name": "loss"},
#         "parameters": {
#             "ce_loss": {"values": [0.1, 0.5, 1, 2]},
#             "chamfer_loss": {"values": [0.1, 0.5, 1, 2]},
#             "laplacian_loss": {"values": [0.1, 0.5, 1, 2]},
#             "normal_consistency_loss": {"values": [0.1, 0.5, 1, 2]},
#             "edge_loss": {"values": [0.1, 0.5, 1, 2]},
#             },
#         }
#     sweep_id = wandb.sweep(sweep_config, entity="reitxel", project="voxel2mesh_v2")
#     wandb.agent(sweep_id, main, count=20)

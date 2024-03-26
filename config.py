import numpy as np
import torch 

from dataset import Crops

class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config(exp_id):
      
    cfg = Config()
    ''' Experiment '''
    cfg.experiment_idx = exp_id 
    cfg.trial_id = None
    
    cfg.save_dir_prefix = 'Experiment_' # prefix for experiment folder
    cfg.name = 'voxel2mesh'

    ''' 
    **************************************** Paths ****************************************
    save_path: results will be saved at this location
    dataset_path: dataset must be stored here.
    '''
    cfg.save_path = './data_crops_results'    # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
    cfg.dataset_path = './' # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
    
    
    # Initialize data object for. 
    # cfg.data_obj = None     # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
    cfg.data_obj = Crops()

    assert cfg.save_path != None, "Set cfg.save_path in config.py"
    assert cfg.dataset_path != None, "Set cfg.dataset_path in config.py"
    assert cfg.data_obj != None, "Set cfg.data_obj in config.py"

    ''' 
    ************************************************************************************************
    ''' 

    ''' Dataset '''  
    # input should be cubic. Otherwise, input should be padded accordingly.
    cfg.patch_shape = (64, 64, 64) 
    #cfg.patch_shape = (32, 32, 32) 

    cfg.ndims = 3
    cfg.augmentation_shift_range = 10

    ''' Model '''
    cfg.first_layer_channels = 16
    cfg.num_input_channels = 1
    cfg.steps = 4

    # Only supports batch size 1 at the moment. 
    cfg.batch_size = 1 

    cfg.num_classes = 2
    cfg.batch_norm = True  
    cfg.graph_conv_layer_count = 4
    cfg.se_block = True
    
    # curvature loss max weight
    cfg.curv_weight_max = 5.0
  
    ''' Optimizer '''
    cfg.learning_rate = 1e-4

    ''' Training '''
    cfg.numb_of_itrs = 300 # 300000
    cfg.eval_every = 100 # 1000 # saves results to disk

    # ''' Reporting '''
    # cfg.wab = True # use weight and biases for reporting
    
    return cfg

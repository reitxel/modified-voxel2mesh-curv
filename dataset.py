# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:50:33 2023

@author: rgonzal2
"""

from data import get_item, normalize_vertices, sample_outer_surface_in_voxel, clean_border_pixels, voxel2mesh, sample_to_sample_plus
from utils.utils_common import DataModes
from utils.metrics import jaccard_index, dice_score, chamfer_weighted_symmetric, chamfer_directed


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import torchio as tio
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sample:
    def __init__(self, x, y, atlas=None):
        self.x = x
        self.y = y
        self.atlas = atlas

class SamplePlus:
    def __init__(self, x, y, y_outer=None, x_super_res=None, y_super_res=None, y_outer_super_res=None, shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res  
        self.shape = shape

class CropsDataset():

    def __init__(self, data, cfg, mode): 
        self.data = data  

        self.cfg = cfg
        self.mode = mode
 

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx] 
        return get_item(item, self.mode, self.cfg)         

class Crops():
    
    def quick_load_data(self, cfg, resize=False):
        data_root = cfg.dataset_path
        data = {}
        down_sample_shape = cfg.patch_shape

        img_suffix = ".nii.gz"
        mask_suffix = "_mask.nii.gz"
        
        for mode in [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]:
            data_dir = os.path.join(cfg.dataset_path, 'data_crops_{}'.format(mode))
            img_files =[filename for filename in os.listdir(data_dir) if filename.endswith(img_suffix) and not filename.endswith(mask_suffix)]
            label_files = [filename.replace(img_suffix, mask_suffix) for filename in img_files] 
            
            samples = []
            for idx in range(len(img_files)):
                img_file = os.path.join(data_dir, img_files[idx])
                label_file = os.path.join(data_dir, label_files[idx])
                
                img = tio.ScalarImage(img_file)
                lbl = tio.ScalarImage(label_file)
                
                if resize:
                    resizet = tio.Resize(64)
                    img = resizet(img)
                    lbl = resizet(lbl)
        
                img = img.as_sitk()
                img_array = sitk.GetArrayFromImage(img)

                lbl = lbl.as_sitk()
                lbl_array = sitk.GetArrayFromImage(lbl)
        
            
                #img_data = np.expand_dims(img_array, axis=0).astype(np.float32)
                # lbl_data = np.expand_dims(lbl_array, axis=0).astype(np.float32)
        
                img_data = img_array
                lbl_data = lbl_array

                
                x = np.float32(img_data) 
                x = torch.from_numpy(x)
                #x = x.to(device)
                # img_tensor = img_tensor.unsqueeze(0)

                y = np.int64(lbl_data)
                y = torch.from_numpy(y)
                #y = y.long()
                #y = y.to(device)
                
                input_shape = x.shape
                scale_factor = (np.max(down_sample_shape)/np.max(input_shape))

                x = F.interpolate(x[None, None], scale_factor=scale_factor, mode='trilinear')[0, 0]
                y = F.interpolate(y[None, None].float(), scale_factor=scale_factor, mode='nearest')[0, 0]#.long()
                y[y != 0] = 1
                y.long()

                samples.append(Sample(x, y)) 
                        
            new_samples = sample_to_sample_plus(samples, cfg, mode)
            data[mode] = CropsDataset(new_samples, cfg, mode)
            
        return data
        
    
    def load_data(self, cfg):
        # Assume cfg has the dataset path, modes and any other configuration needed
        data = {}
        for mode in [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]:
            data_dir = os.path.join(cfg.dataset_path, 'data_crops_{}'.format(mode))  # Assuming mode value corresponds to directory name
            data[mode] = MyOwnDataset(data_dir, cfg, mode)
        return data
    
                
    def evaluate(self, target, pred, cfg):
        results = {}


        if target.voxel is not None: 
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard
            
            # include dice score computation
            val_dice = dice_score(target.voxel, pred.voxel, cfg.num_classes)
            results['dice'] = val_dice

        if target.mesh is not None:
            target_points = target.points
            pred_points = pred.mesh
            val_chamfer_weighted_symmetric = np.zeros(len(target_points))

            for i in range(len(target_points)):
                val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i].cpu(), pred_points[i]['vertices'])

            results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value):

        key = 'jaccard'
        # uncomment this line for testing and comment the following one
        # new_value = new_value[DataModes.TESTING][key]
        new_value = new_value[DataModes.VALIDATION][key]

        if best_so_far is None:
            return True
        else:
            # uncomment this line for testing and comment the following one
            # best_so_far = best_so_far[DataModes.TESTING][key]
            best_so_far = best_so_far[DataModes.VALIDATION][key]
            return True if np.mean(new_value) > np.mean(best_so_far) else False

    

    

    

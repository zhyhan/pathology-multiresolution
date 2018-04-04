#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:56:57 2018

@author: zhongyi
"""
from api import config_fun
# from api import prob_map_fcn as prob_map
from api import prob_map_fcn_kb as prob_map
# from api import prob_map_cls as prob_map
from api import heat_map_fun
from api import slide_fun
import openslide
import numpy as np
#import train_helper
import os
import torch
from models import custom_net
from models import vgg
#from PIL import Image
# import matplotlib.pyplot as plt


def _get_final_level(img, level):
        
        if img.level_count-1 >= level:
            return level
            
        elif img._img.level_count-1 == 1:
            raise ValueError('%s only has one level resolutional image!')
        else:
            level = img.level_count-1      
            return level
        
def low_resolution_test(cfg, num_classes, file_name, level):
    """
    Input: patches or filename;
    Output: patch_coordinates.
    
    """
    #lr reprents the low resolution model; hr: high resolution.
    if cfg.patch_size == 64:
        model_lr = custom_net.net_64(num_classes = num_classes)
    elif cfg.patch_size == 32:
        model_lr = custom_net.net_32(num_classes = num_classes)
    else:
        print('Do not support present patch size %s'%cfg.patch_size)
    #model_lr = custom_net.LeNet(num_classes = cfg.num_classes)
    checkpoint_lr = torch.load(cfg.init_model_file)
    model_lr.load_state_dict(checkpoint_lr['model_param'])
    model_lr.cuda()
    model_lr = torch.nn.DataParallel(model_lr, device_ids=cfg.gpu_id)
    model_lr.eval()
    
    
    raw_img, sorted_abnormal_patches, time, patch_out_size, b_map, p_map = prob_map.patch_filter(cfg, model_lr, file_name, level)
    print('The number of abnormal patches is %d'%len(sorted_abnormal_patches))
    
    return raw_img, sorted_abnormal_patches, time, patch_out_size, b_map, p_map

#def check_low_resolution_test(cfg, num_classes, file_name, level):
#    """
#    Input: patches or filename;
#    Output: patch_coordinates.
#    
#    """
#    #lr reprents the low resolution model; hr: high resolution.
#    model_lr = custom_net.LeNet(num_classes = cfg.num_classes)
#    checkpoint_lr = torch.load(cfg.init_model_file)
#    model_lr.load_state_dict(checkpoint_lr['model_param'])
#    model_lr.cuda()
#    model_lr = torch.nn.DataParallel(model_lr, device_ids=cfg.gpu_id)
#    model_lr.eval()
#    
#    raw_img, b_map, p_map = prob_map.check_patch_filter(cfg, model_lr, file_name, level)
#    #print('The number of abnormal patches is %d'%len(sorted_abnormal_patches))
#    
#    return raw_img, b_map, p_map


def highest_resolution_test(cfg, file_name, sorted_abnormal_patches, time, patch_out_size, b_map, p_map):
    """
    Input: patch_coordinates;
    Output: heat map.
    
    """
    model_hr = vgg.vgg19_bn(pretrained = False, num_classes = num_classes)
    checkpoint_hr = torch.load(checkpoint_path_hr)
    model_hr.load_state_dict(checkpoint_hr['model_param'])
    model_hr.cuda()
    model_hr = torch.nn.DataParallel(model_hr, device_ids=cfg.gpu_id)
    model_hr.eval()
    b_map, p_map = prob_map.generate_prob_map_hr(cfg, file_name, sorted_abnormal_patches, model_hr, time, patch_out_size, b_map, p_map)
    return b_map, p_map



if __name__ == '__main__': 
    cfg = config_fun.config()
    num_classes = cfg.num_classes    
    checkpoint_path_hr = cfg.checkpoint_pth_file_level0
    optim_state_file =  cfg.optim_state_file_level0
    f = open(cfg.test_file, 'r')
    for s in f.readlines():
        if s.split() == []: continue
        # file_name, label = s.split('*')
        file_name = s.split('\n')[0]
        print(file_name)  
        #if os.path.exists(file_name + '.heatmap.jpg'):
        #    continue
        
        try:
            data = slide_fun.AllSlide(file_name)
            level =_get_final_level(data, cfg.target_level)
            level_size = data.level_dimensions[level]
            image = data.read_region((0, 0), level, level_size)
            image.close()
            
        except openslide.OpenSlideUnsupportedFormatError:
            # print(file_name, 'unsupported error')
            continue
        except openslide.lowlevel.OpenSlideError:
            # print(file_name, 'low level error')
            continue
        print('processing ' + file_name)
        
        #raw_img, b_map, p_map = check_low_resolution_test(cfg, num_classes, file_name, level)
        
        raw_img, sorted_abnormal_patches, time, patch_out_size, b_map, p_map = low_resolution_test(cfg, num_classes, file_name, level)
        b_map, p_map = highest_resolution_test(cfg, file_name, sorted_abnormal_patches, time, patch_out_size, b_map, p_map)
        save_dir_pre = os.path.join(cfg.gm_foder,
                                os.path.basename(file_name).split('.')[0])
        raw_img_dir = save_dir_pre + '_raw_img' + cfg.img_ext
        h_map_img_dir = save_dir_pre + '_h_map_img' + cfg.img_ext
        p_map_npy_dir = save_dir_pre + '_p_map_img' + '.npy'
    
        np.save(p_map_npy_dir, p_map)
        np.save(file_name+'.pmap.npy', p_map)
    
        raw_img.save(raw_img_dir)
        raw_img.save(file_name + '.raw.jpg')
        raw_img.close()
        htmap_img = heat_map_fun.get_heatmap_from_prob(p_map)
        htmap_img.save(h_map_img_dir)
        #htmap_img.save(file_name + '.heatmap.jpg')
        htmap_img.close() 
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 19:45:37 2018

@author: zhongyi
"""
#import torch
import torch.nn as nn
import torch.nn.functional as F


#class custom_net(nn.Module):
#    def __init__(self, num_classes = 2):
#        super(custom_net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 64, 5)
#        self.batch_norm_1 = nn.BatchNorm2d(64)
#        self.conv2 = nn.Conv2d(64, 128, 3)
#        self.batch_norm_2 = nn.BatchNorm2d(128)
#        self.conv3 = nn.Conv2d(128, 256, 3)
#        self.batch_norm_3 = nn.BatchNorm2d(256)
#        self.conv4 = nn.Conv2d(256, 256, 3, dilation=2, padding=1)
#        self.batch_norm_4 = nn.BatchNorm2d(256)
#        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
#        self.batch_norm_5 = nn.BatchNorm2d(256)
#        self.conv6 = nn.Conv2d(256, 256, 3, dilation=2, padding=1)
#        self.batch_norm_6 = nn.BatchNorm2d(256)
#        self.conv8 = nn.Conv2d(256, 512, 3, padding=0)
#        self.batch_norm_8 = nn.BatchNorm2d(512)
#        self.conv9 = nn.Conv2d(512, 1024, 1, padding=1)
#        self.batch_norm_9 = nn.BatchNorm2d(1024)
#        self.conv10 = nn.Conv2d(1024, 512, 1, padding=1)
#        self.batch_norm_10 = nn.BatchNorm2d(512)
#        self.conv11 = nn.Conv2d(512, num_classes, 1)
#        
#        #self._initialize_weights()
#        
#    def forward(self, x):
#        out = self.batch_norm_1(F.relu(self.conv1(x)))
#        print(out.shape)
#        out = self.batch_norm_2(F.relu(self.conv2(out)))
#        print(out.shape)
#        out = F.max_pool2d(out, 2)
#        out = self.batch_norm_3(F.relu(self.conv3(out)))
#        print(out.shape)
#        out = self.batch_norm_4(F.relu(self.conv4(out)))
#        print(out.shape)
#        out = self.batch_norm_5(F.relu(self.conv5(out)))
#        print(out.shape)
#        out = self.batch_norm_6(F.relu(self.conv6(out)))
#        print(out.shape)
#        #out = self.batch_norm_7(F.relu(self.conv7(out)))
#       # out = torch.add(out_a1, out_a2)
#        out = self.batch_norm_8(F.relu(self.conv8(out)))
#        out = self.batch_norm_9(F.relu(self.conv9(out)))      
#        out = self.batch_norm_10(F.relu(self.conv10(out)))
#        out = self.conv11(out)
#        print(out.shape)
#        return out

class net_32(nn.Module):
    def __init__(self, num_classes = 2):
        super(net_32, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
class net_64(nn.Module):
    def __init__(self, num_classes = 2):
        super(net_64, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.batch_norm_1 = nn.BatchNorm2d(6)
        
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.batch_norm_2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 64, 3, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.batch_norm_4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.batch_norm_5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.batch_norm_6 = nn.BatchNorm2d(512)
        
        self.conv7 = nn.Conv2d(512, 256, 1)
        
        self.batch_norm_7 = nn.BatchNorm2d(256)
        
        self.conv8 = nn.Conv2d(256, num_classes, 1)
        #self.dropout = nn.Dropout(p=0.5, inplace=False)
       # self.batch_norm_7 = nn.BatchNorm2d(1024)
       # self.fc2   = nn.Linear(1024, 512)
       # self.batch_norm_8 = nn.BatchNorm2d(512)
        #self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.batch_norm_1(F.relu(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.batch_norm_2(F.relu(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = self.batch_norm_3(F.relu(self.conv3(out)))
        out = F.max_pool2d(out, 2)
        out = self.batch_norm_4(F.relu(self.conv4(out)))
        out = self.batch_norm_5(F.relu(self.conv5(out)))
        out = self.batch_norm_6(F.relu(self.conv6(out)))        
        out = self.batch_norm_7(F.relu(self.conv7(out)))

        #out = self.batch_norm_8(F.relu(self.fc2(out)))
        
        out = self.conv8(out)
        return out
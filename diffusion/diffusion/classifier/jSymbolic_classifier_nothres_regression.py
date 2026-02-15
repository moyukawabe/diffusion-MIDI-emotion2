'''
main!!
分類器処理 : MIDIトークン → 音楽属性値 の回帰を行うコード 
'''

## jSymbolic_feature.py
import shutil
import pandas as pd
import torch
import json
import os
import subprocess
from tqdm import tqdm
import time
import xmltodict
import random
import numpy as np
#import autograd.numpy as np
#from autograd import grad, elementwise_grad
from multiprocessing import Process, Pool, Manager, Lock
import json
from emogen.jSymbolic_lib.jSymbolic_util import read_pitch_feature, read_all_feature
import xmltodict
import subprocess
from functools import partial
import argparse
import torch.nn.functional as F

command_prefix = "java -Xmx6g -jar ./emogen/jSymbolic_lib/jSymbolic_2_2_user/jSymbolic2.jar -configrun ./emogen/jSymbolic_lib/jSymbolic_2_2_user/jSymbolicDefaultConfigs.txt"

## gen_data.py
import multiprocessing
import sys
sys.path.append("..")
import miditoolkit
import numpy.random
import collections, pickle, shutil
from MidiProcessor.midiprocessor import MidiEncoder, midi_utils, MidiDecoder
from tqdm import tqdm
import gc
import math
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge , LinearRegression
import joblib


import matplotlib.pyplot as plt
from sklearn import datasets # 学習用のサンプルデータ
from torch.utils.data import DataLoader # データを整理する
from torch import nn # 全結合層と活性化関数
from torch.nn import functional as F
from torch import optim # 損失関数と最適化関数

# 256 → 100 の回帰モデル
class JSymbolicNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 全結合層を3つ
        self.fc1 = nn.Linear(256, 256).cuda()#
        self.fc2 = nn.Linear(256, 100).cuda()#
        
        # 損失関数と最適化関数
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss().cuda()
        self.optimizer = optim.Adam(self.parameters(),0.01)
        self.dropout = nn.Dropout(p=0.2).cuda()
        self.bn1 = nn.BatchNorm1d(256, track_running_stats=True).cuda()
        self.bn2 = nn.BatchNorm1d(100, track_running_stats=True).cuda() #100   

    def forward(self, x):  # batch_size = 256 , nn.MSELoss(), optim.Adam(self.parameters(),0.0001)
        x = self.fc1(x) #   = nn.Linear(256, 256)
        x = self.bn1(x) #   = nn.BatchNorm1d(256, track_running_stats=True)
        #x = self.dropout(x) 
        x = F.leaky_relu(x) #x = F.relu(x) 
        x = self.fc2(x) 
        x = F.leaky_relu(x)
        return x
    
    
# jSymbolicの処理
class JSymbolic_classifier_nothres_regression():  
    # 音楽属性値を回帰予測　(処理の過程で閾値は使用しない)  
    def get_jSymbolic_feature(self, MIDI, args):
        # nn.Linear
        model = args.music_attribute
        features = []
        i = 0
        print('MIDI   = ' + str(MIDI[:,:,1].shape))
        for i in range(len(MIDI[0,0,:])):
            features.append(model(MIDI[:,:,i].cuda()))
            i = i + 1
        MIDImusic_attribute = torch.stack(features, dim = 2)
        print('music_attribute requires_grad = ' + str(MIDImusic_attribute.requires_grad))
        return MIDImusic_attribute 

    # 閾値の関数 
    def binarize_command(self, command, thresholds, args):
        thresholds_rep= (torch.from_numpy(thresholds.T).unsqueeze(2)).repeat(args.batch_size ,1, args.in_channel)  # 閾値を複製
        # データの (thresholds_weight0[0,:]) を使用して、重みづけ
        thresholds_weight0 = torch.load('./emogen/std_mean_train_a18_2.pt').cuda() # './emogen/std_mean.pt'
        thresholds_weight = torch.where(thresholds_weight0[0,:] == 0.0, (torch.tensor(0.0).float()).cuda(), ((1.0 / thresholds_weight0[0,:]).float()).cuda()) 
        thresholds_weight = torch.unsqueeze(torch.unsqueeze(thresholds_weight,dim=0) , dim=2).repeat(args.batch_size ,1, args.in_channel) 
        # 閾値処理--------
        one_zero =  (command - thresholds_rep.cuda()) * thresholds_weight.cuda() # + 重み：infill_util.py
        print(one_zero.device)
        one_zero_norm = one_zero.cuda() 
        return one_zero_norm #[64,100,32]
    
    '''
    def change_01(relu_value):
        if relu_value == 0.0
            return 0.0  # 0で割る場合は0を返す
        else:
            return 1.0 / relu_value
    '''
    
    # 音楽属性値を0,1の値に変換
    def gen_split_data(self,  music_attribute, arg):
        feature_index = np.load("./emogen/data/feature_index.npy", allow_pickle=True)
        thresholds = np.load("./emogen/data/threshold.npy", allow_pickle=True)
        j = 0
        for i in range(1):
            try:
                no_jS_feature = music_attribute
            except:
                continue
            '''''     
            if(jS_feature.size == 0):
                jS_feature =torch.zeros(1495)
            else:
                      #if len(jS_feature) != 1495:
                if jS_feature[0].size != 1495:
                    continue
            ''''' 
            split_command = self.binarize_command(no_jS_feature, thresholds, arg) # 0,1の値に変換
        return split_command #lossで比較するもの

    np.random.seed(42)
    random.seed(42)


# gaussian_diffusion.py line 1306~
    def main(self, midi_sampled, arg):
        parser = argparse.ArgumentParser()
        music_attribute = self.get_jSymbolic_feature(midi_sampled, arg) 
        attribute_sampleds = self.gen_split_data(music_attribute,arg) # 音楽属性値を100個
        print('attribute_sampleds = ' + str(attribute_sampleds))
        print('attribute_sampleds requires_grad = ' + str(attribute_sampleds.requires_grad))
        

        return attribute_sampleds 


if __name__ == "__main__":
    jsmbolic = JSymbolic_classifier_nothres_regression()
    jsmbolic.main("./emogen/data/Piano")







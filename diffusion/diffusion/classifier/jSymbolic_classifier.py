'''
分類器処理 : MIDIトークン → 音楽属性値 の回帰を行うコード (before)
'''
#jSymbolic_feature.py
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
from multiprocessing import Process, Pool, Manager, Lock
import json
from emogen.jSymbolic_lib.jSymbolic_util import read_pitch_feature, read_all_feature
import xmltodict
import subprocess
from functools import partial
import argparse

command_prefix = "java -Xmx6g -jar ./emogen/jSymbolic_lib/jSymbolic_2_2_user/jSymbolic2.jar -configrun ./emogen/jSymbolic_lib/jSymbolic_2_2_user/jSymbolicDefaultConfigs.txt"

#gen_data.py
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
import joblib


class JSymbolic_classifier():
    
    #jSymbolic_feature.py
    def rename_midi_path(self, root):
        midi_name_list = os.listdir(root)
        for midi_name in midi_name_list:
            os.rename(root + "/" + midi_name, root + f"/{midi_name.replace(' ', '_')}")
        
    def get_jSymbolic_feature(self, file, midi_name, root):
        #midi_name = file_name[:-4]  # target model_output
        file.dump(root + f"/midi/{midi_name}.mid")
        midi_path = root + f"/midi/{midi_name}.mid"
        #midi_path = midi_name
        #path = os.path.join(root, "feature/" + f"{midi_name.replace(' ', '_')}.xml")
        path = os.path.join(root, "feature/" + f"{midi_name}.xml")
        # print(midi_path)
        #if os.path.exists(path):
           #return 0
        #if not os.path.exists(path):
        #if midi_name = "x_start_emo"
           #return 0
        #else
        new_command = " ".join([command_prefix, midi_path, path, 
                                "./test_def.xml"])
        os.system(new_command)
        return 0


    #gen_data.py

    def binarize_data(self, midi_sampled, path_root):
        save_root = path_root + midi_sampled + "/data-bin"
        dict_path = save_root + f"/dict.txt"
        command = f"fairseq-preprocess --only-source --destdir {save_root} --srcdict {dict_path} "\
                f"--validpref {path_root}/valid.txt  --testpref {path_root}/test.txt  --trainpref {path_root}/train.txt --workers 4 "
        text = os.popen(command).read()
        print(text)


    def binarize_command(self, command, thresholds):
        discrete_feature = []
        for k in range(command.shape[0]):
           # print(command.shape[0])
            #print(k)
            thres = thresholds[k]
            discrete_feature.append(np.searchsorted(thres, command[k]))
            #print(discrete_feature)
        return discrete_feature
    
    def gen_split_data(self, midi_sampled, filename, remi_sampled, path_root):
        feature_index = np.load("./emogen/data/feature_index.npy", allow_pickle=True)
        #all_feature_name = np.load("./emogen/data/all_feature_name.npy", allow_pickle=True)
        #print(all_feature_name)
        ##feature_index = np.load(feature_index, allow_pickle=True)
        thresholds = np.load("./emogen/data/threshold.npy", allow_pickle=True)
        ##thresholds = np.load(thresholds, allow_pickle=True)
        thresholds2 = torch.tensor(thresholds) ##
        #print(thresholds2) ##
        #print(feature_index)

        save_root = path_root
        os.makedirs(save_root, exist_ok=True)
        fn_list = os.listdir(path_root + "/remi")
        random.shuffle(fn_list)

        
        ##for split in ["train", "valid", "test"]:
           ## split_command = []
           ## if split == "train":
            ##    s,e = 0, int(len(fn_list)*0.8)
            ##elif split == "valid":
            ##    s,e = int(len(fn_list)*0.8), int(len(fn_list)*0.9)
            ##else:
            ##    s,e = int(len(fn_list)*0.9), len(fn_list)
        
        
            ##with open(path_root + f"/{split}.txt", "w") as split_txt:
        with open(path_root + f"/{filename}.txt", "w") as midi_sampled_txt:
              ##  split_fn_list = []
               j = 0
               split_command = []
               ## for i, fn in enumerate(tqdm(fn_list[s:e])):
               for i , fn in enumerate(remi_sampled):
                   ## fn_name = fn.split(".")[0]
                  try:
                        ##jS_feature = read_all_feature(path_root + f"/feature/{fn_name}.xml")
                     jS_feature = read_all_feature(path_root + f"/feature/{filename}.xml")
                  except:
                     continue
                  jS_feature = np.array(jS_feature)
                  if len(jS_feature) != 1495:
                        continue
                  jS_feature = np.array(jS_feature)
                  binary_command = self.binarize_command(jS_feature[feature_index], thresholds)
                  #print(jS_feature[feature_index])
                  #split_command = []
                  split_command.append(binary_command)
                  #print(binary_command)
                    ##split_fn_list.append(fn)
                    ##with open(path_root + f'/remi/{fn}', "r") as f:
                        ##remi_tokens = f.read().strip("\n").strip(" ")
         #remi_tokens = remi_sampled
                        ##split_txt.write(remi_tokens + "\n")
         #midi_sampled_txt.write(remi_tokens + "\n")
                  j += 1
        split_command = np.array(split_command)
                ##print(split_command) ##
                ##np.save(save_root + f"/{split}_fn_list.npy", split_fn_list)
                ##np.save(save_root + f'/{split}_command.npy', split_command)
         #np.save(save_root + f'/{midi_sampled}_command.npy', split_command)
                ##assert len(split_fn_list) == len(split_command), "length dismatch!"

        return split_command #lossで比較するもの




    np.random.seed(42)
    random.seed(42)


# gaussian_diffusion.py line 1306~
    def main(self, midi_sampled, filename, remi_sampled):
        parser = argparse.ArgumentParser()
        data_path = "./emogen/data/Piano"
        #midi_sampled = os.listdir(data_path + f"/midi")   ## target, model_output
        os.makedirs(data_path +f"/feature", exist_ok=True)
        self.get_jSymbolic_feature(midi_sampled, filename, data_path)
        #with Pool(processes=8) as pool:
            #result = iter(tqdm(pool.imap(partial(self.get_jSymbolic_feature, root = data_path), midi_sampled, filename),
                              # total=len(midi_sampled)))
            #for i in range(len(midi_sampled)):
                 #try:
                #next(result)

        midi_sampleds = self.gen_split_data(midi_sampled, filename, remi_sampled, data_path)
        #self.binarize_data(midi_sampled, data_path)

        return midi_sampleds





if __name__ == "__main__":
    jsmbolic = JSymbolic_classifier()
    jsmbolic.main("./emogen/data/Piano")







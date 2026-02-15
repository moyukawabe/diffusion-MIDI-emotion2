'''
# 複数の音楽属性値取得 → csvファイルなどに書き込みたい場合
# できれば、GPUを使用 (cudf)
# featureというファイルをあらかじめ作っておく
'''
import shutil

import pandas as pd
#import cudf
import json
import os
import subprocess
from tqdm import tqdm
import time
import xmltodict
import random
import numpy as np
from multiprocessing import Process, Pool, Value, Array,Manager
import json
from jSymbolic_util import read_pitch_feature, read_all_feature
import xmltodict
import subprocess
from functools import partial
import argparse

# csv変換(エラー対策)
#import jpype
#import asposecells 
#jpype.startJVM()
#from asposecells.api import Workbook

command_prefix = "java -Xmx6g -jar ./jSymbolic_2_2_user/jSymbolic2.jar -configrun ./jSymbolic_2_2_user/jSymbolicDefaultConfigs.txt"
#list = []


def rename_midi_path(root):
    midi_name_list = os.listdir(root)
    for midi_name in midi_name_list:
        os.rename(root + "/" + midi_name, root + f"/{midi_name.replace(' ', '_')}")

def get_jSymbolic_feature(file_name, list, list2, root):
    midi_name = file_name[:-4]          
    print("file0 = " +str(file_name))
    print("file = " +str(midi_name)) 
    midi_path = root + f'./{midi_name}.mid' 
    #感情→音楽属性値　

    path = os.path.join(root, f"feature/{midi_name}.xml") # featureというファイルをあらかじめ作っておく
    new_command = " ".join([command_prefix, midi_path, path,
                                "./test_def.xml"])
    os.system(new_command)
    
    #csvファイルが存在しない時
    if os.path.exists(f"{path[:-4]}.csv"):
        print('True')
        list.append(pd.read_csv(f'{path[:-4]}.csv',encoding="CP932",dtype = 'object'))
        os.remove(path[:-4] + '.csv')
        os.remove(path[:-4] + '.xml') #os.path.splitext(path)[0] + '.xml')
        os.remove(path[:-4] + '.arff') 
    else:
        print("False")  # csvファイルできなかったやつ
        # list2.append(path[-15:-4])
        list2.append(midi_name)
        print(list2) 
        os.remove(path[:-4] + '.xml')
    #'''
    return 0

np.random.seed(42)
random.seed(42)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    manager =  Manager()
    list = manager.list()
    list2 = manager.list()
    j =  manager.Value('i', 0)
    data_path = "/home/moyu/program/Diffusion-LM-on-Symbolic-Music-Generation/improved-diffusion/generation_CFG/18_2_re_DEAM/"  
    midi_sampled = os.listdir(data_path + f"18_2_2weight_DEAM_Q1/")
    
    df_list = np.zeros((0,1496))
    with Pool(processes=1) as pool:
        result = iter(tqdm(pool.imap(partial(get_jSymbolic_feature, list = list, list2 = list2 , root = data_path), midi_sampled),
                           total= int(len(midi_sampled))))
        i = 0
        for i in range(int(len(midi_sampled))):
            next(result) 
            
            # csvファイル
            df_list = pd.DataFrame(np.squeeze(np.array(list)))#(リストの)データフレーム作成
            df_list.to_csv('output_attribute/DLM_traindata_attribute1.csv',index=False,encoding="utf-8")
            '''
            # GPU使用する場合
            cudf_list = cudf.from_pandas(df_list) # データ(リストを)GPUに載せる
            cudf_read = cudf.concat(cudf_list)
            cudf_list.to_csv('giant_midi_piano__train_split_attribute.csv',index=False,encoding="utf-8") # 変更 csv:音楽属性値まとめ
            '''
            #ファイル削除
            # if(os.path.exists(data_path + 'feature' + str(j.value - 1))):
                #shutil.rmtree(data_path + 'feature'+ str(j.value - 1))
            j.value += 1
    f = open('output_attribute/DLM_traindata_attribute.txt', 'a', encoding='UTF-8') # 変更　txt:音楽属性知取り出せてないmidi音楽
    f.write(str(list2))
    f.close()
    print("end")
 
    
    
    








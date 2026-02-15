'''
# EMOPIA_cls
# 各音楽のValence/ArousalのHigh/Low値を推定
# まとめて推定値を取る
#  nohup python inference_many.py --types remi --task arousal --file_path '/.npz' > validdataArousal.txt &
'''

import os
import json
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace

import glob
import torch
import numpy
#import torchaudio
from omegaconf import DictConfig, OmegaConf
#from audio_cls.src.model.net import ShortChunkCNN_Res
from midi_cls.src.model.net import SAN
from midi_cls.midi_helper.remi.midi2event import analyzer, corpus, event
from midi_cls.midi_helper.magenta.processor import encode_midi
# 書き込み
import csv

path_data_root = "./midi_cls/midi_helper/remi/"
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')
midi_dictionary = pickle.load(open(path_dictionary, "rb"))
event_to_int = midi_dictionary[0]


# wav
def torch_sox_effect_load(mp3_path, resample_rate):
    effects = [
        ['rate', str(resample_rate)]
    ]
    waveform, source_sr = torchaudio.load(mp3_path)
    if source_sr != 22050:
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
    return waveform

# midi_remi
def remi_extractor(midi_path, event_to_int):
    midi_obj = analyzer(midi_path)
    song_data = corpus(midi_obj)
    event_sequence = event(song_data)
    quantize_midi = []
    for i in event_sequence:
        try:
            quantize_midi.append(event_to_int[str(i['name'])+"_"+str(i['value'])])
        except KeyError:
            if 'Velocity' in str(i['name']):
                quantize_midi.append(event_to_int[str(i['name'])+"_"+str(i['value']-2)])
            else:
                continue #skip the unknown event
    return quantize_midi

# midi_magenta
def magenta_extractor(midi_path):
    return encode_midi(midi_path)

# データ作成のため、推定値をまとめて取得
def predict(args) -> None:
    device = args.cuda if args.cuda and torch.cuda.is_available() else 'cpu'
    config_path = Path("best_weight", args.types, args.task, "hparams.yaml")
    checkpoint_path = Path("best_weight", args.types, args.task, "best.ckpt")
    config = OmegaConf.load(config_path)
    label_list = list(config.task.labels)
    if args.types == "wav":
        model = ShortChunkCNN_Res(
                sample_rate = config.wav.sr,
                n_fft = config.hparams.n_fft,
                f_min = config.hparams.f_min,
                f_max = config.hparams.f_max,
                n_mels = config.hparams.n_mels,
                n_channels = config.hparams.n_channels,
                n_class = config.task.n_class
        )
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
    else: # こちらを使用
        model = SAN( 
            num_of_dim= config.task.num_of_dim, 
            vocab_size= config.midi.pad_idx+1, 
            lstm_hidden_dim= config.hparams.lstm_hidden_dim, 
            embedding_size= config.hparams.embedding_size, 
            r= config.hparams.r)
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu')) #'cpu'
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
    model = model.to('cpu') 
    
## フォルダ読み込み
    # files = glob.glob(args.file_path + '/*.mid')  #files = os.listdir(args.file_path).glob("/*.mid")
    files = numpy.load(args.file_path)['arr_0'] 
    #l = []
    Highcount = 0
    ##header = ['', 'file', 'emotion', 'Inference values']
    ##index = []
    ##index.append(h)
    pred_labels = [] #numpy.full(8, 'aa')
    pred_values = [] #numpy.empty((8,2))
    for j, file in enumerate(files):  # #for i in range(16):
            f = file # args.file_path # f = os.path.join(args.file_path, file)
            if args.types == "midi_like": # こちらを使用
                quantize_midi = magenta_extractor(f) 
                model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
                prediction = model(model_input.to('cpu')) 
            elif args.types == "remi":
                quantize_midi = f
                model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
                prediction = model(model_input.to('cpu')) 
            elif args.types == "wav":
                model_input = torch_sox_effect_load(file , 22050).mean(0, True) #
                sample_length = config.wav.sr * config.wav.input_length
                frame = (model_input.shape[1] - sample_length) // sample_length
                audio_sample = torch.zeros(frame, 1, sample_length)
                for i in range(frame):
                    audio_sample[i] = torch.Tensor(model_input[:,i*sample_length:(i+1)*sample_length])
                prediction = model(audio_sample.to('arg.cuda')) # 'cpu' →　args.cuda
                prediction = prediction.mean(0,False)

            pred_label = label_list[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()]
            pred_value = prediction.squeeze(0).detach().cpu().numpy()
            pred_labels.append(pred_label)
            pred_values.append(pred_value)
            if pred_label == 'HA': # 'HV':
                Highcount = Highcount + 1 # Highのものをカウント
            # pred_labels[j] = pred_label
            # pred_values[j] = pred_value
            print(j)
    pred_labels_np = numpy.array(pred_labels)
    pred_values_np = numpy.array(pred_values)
    # numpy.savez('validdata_valence.npz', a=pred_labels_np, b=pred_values_np) #　ファイル保存
    numpy.savez('validdata_Arousal.npz', a=pred_labels_np, b=pred_values_np)
    return Highcount #pred_labels_np, pred_values_np
    '''   # CSVファイルへの書き込み
    ##with open('./dataset/sample_data/emotion_valence.csv', 'w') as f: 
        ##writer = csv.writer(f)
        ##writer.writerow(header)
        ##for i, row in zip(index, l):
            ##writer.writerow([i] + row)
    ##return pred_label, pred_value
    '''


# evaluate > Emotion.ipynbで使用
def predict_evaluate(cuda,types,task,file_path) -> None:
    device = cuda if cuda and torch.cuda.is_available() else 'cpu'
    config_path = Path("best_weight", types, task, "hparams.yaml")
    checkpoint_path = Path("best_weight", types, task, "best.ckpt")
    config = OmegaConf.load(config_path)
    label_list = list(config.task.labels)
    if types == "wav":
        model = ShortChunkCNN_Res(
                sample_rate = config.wav.sr,
                n_fft = config.hparams.n_fft,
                f_min = config.hparams.f_min,
                f_max = config.hparams.f_max,
                n_mels = config.hparams.n_mels,
                n_channels = config.hparams.n_channels,
                n_class = config.task.n_class
        )
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
    else: # こちらを使用
        model = SAN( 
            num_of_dim= config.task.num_of_dim, 
            vocab_size= config.midi.pad_idx+1, 
            lstm_hidden_dim= config.hparams.lstm_hidden_dim, 
            embedding_size= config.hparams.embedding_size, 
            r= config.hparams.r)
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu')) #'cpu'
        new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
        new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
        model.load_state_dict(new_state_dict)
        model.eval()
    model = model.to('cpu') #arg.cuda')
    
## フォルダ読み込み
    files = os.listdir(file_path)#.glob("/*.mid")
    Highcount = 0
    pred_labels = []
    pred_values = []
    for j, file in enumerate(files):  
        if  file.endswith(".mid") :  
            f = f'{file_path}/{file}' 
            if types == "midi_like": # こちらを使用
                quantize_midi = magenta_extractor(f) 
                model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
                if model_input.tolist() != [[]]:
                    prediction = model(model_input.to('cpu')) 
                else:
                    prediction = torch.zeros(1,2)
            elif types == "remi":
                quantize_midi = f
                model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
                prediction = model(model_input.to('cpu')) #arg.cuda'))
            elif types == "wav":
                model_input = torch_sox_effect_load(file , 22050).mean(0, True) 
                sample_length = config.wav.sr * config.wav.input_length
                frame = (model_input.shape[1] - sample_length) // sample_length
                audio_sample = torch.zeros(frame, 1, sample_length)
                for i in range(frame):
                    audio_sample[i] = torch.Tensor(model_input[:,i*sample_length:(i+1)*sample_length])
                prediction = model(audio_sample.to('arg.cuda')) # 'cpu' →　args.cuda
                prediction = prediction.mean(0,False)

            pred_label = label_list[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()]
            pred_value = prediction.squeeze(0).detach().cpu().numpy()
            pred_labels.append(pred_label)
            pred_values.append(pred_value)
            if pred_label == 'HA' or pred_label =='HV':
                Highcount = Highcount + 1 # 推定がHighのものをカウント
            # pred_labels[j] = pred_label
            # pred_values[j] = pred_value
    pred_labels_np = numpy.array(pred_labels)
    pred_values_np = numpy.array(pred_values)
    # numpy.savez('validdata_valence.npz', a=pred_labels_np, b=pred_values_np) #　ファイル保存
    return Highcount , pred_labels_np, pred_values_np



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--types", default="midi_like", type=str, choices=["midi_like", "remi", "wav"])
    parser.add_argument("--task", default="ar_va", type=str, choices=["ar_va", "arousal", "valence"])
    parser.add_argument("--file_path", default="./dataset/sample_data/Sakamoto_MerryChristmasMr_Lawrence.mid", type=str)
    # file_path = '.npz'
    parser.add_argument('--cuda', default='cuda:0,1', type=str)
    args = parser.parse_args()
    # _, _ = predict(args)
    Highcount = predict(args)
    # pred_label, pred_value = predict(args)
    print(Highcount)
    # print(str(pred_value[:,0].mean()) +' , '+ str(pred_value[:,1].mean()) +',' + str(pred_label))
    #_ = predict(args)

'''
学習用データセットを用意！
'''

import random

import torch
from miditoolkit import MidiFile
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List

from improved_diffusion.text_datasets import _collate_batch_helper
from improved_diffusion.utils import point_debug
from symbolic_music.advanced_padding import advanced_remi_bar_block
from symbolic_music.utils import get_tokenizer


def __create_embedding_model(data_args, vocab_size):
    model = torch.nn.Embedding(vocab_size, data_args.in_channel)  # in_channel: embedding dim
    print('initializing the random embeddings', model)
    torch.nn.init.normal_(model.weight)
    path_save = f'{data_args.checkpoint_path}/random_emb.torch'
    print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
    torch.save(model.state_dict(), path_save)
    return model


def __padding(data_args, tokens_list, block_size) -> List[List[int]]:
    """
    block padding, will make blocks for long examples. note that [s] and [end] might be lost in many many tracks
    """
    if data_args.padding_mode == 'bar_block':
        return advanced_remi_bar_block(tokens_list, block_size)
    if data_args.padding_mode == 'block':
        print('using block padding')
        concatenated_tokens = sum(tokens_list, [])
        total_length = (len(concatenated_tokens) // block_size) * block_size
        print(f'total length: {total_length}')
        return [concatenated_tokens[i: i + block_size] for i in range(0, total_length, block_size)]
    if data_args.padding_mode == 'pad':
        print('using pad padding')
        tokens_list = _collate_batch_helper(tokens_list, 0, block_size)
        return tokens_list
    raise NotImplementedError


class MidiDataset(Dataset):
    def __init__(
            self, midi_data_list, resolution, data_args, model_arch, eigen_transform=None,
            mapping_func=None, model_emb=None
    ):
        super().__init__()
        self.resolution = resolution
        self.midi_data_list = midi_data_list
        self.length = len(self.midi_data_list)
        self.model_arch = model_arch
        self.data_args = data_args
        print(self.resolution)
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arr = np.array(self.midi_data_list[idx]['hidden_states'], dtype=np.float32)
        if self.eigen_transform is not None:
            old_shape = arr.shape
            # arr = arr.reshape(1, -1) @ self.eigen_transform
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)

        if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
            arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

        out_dict = {'input_ids': np.array(self.midi_data_list[idx]['input_ids'])}
        if self.data_args.experiment_mode == 'conditional_gen':  # TODO not implementing conditional gen for now
            out_dict['src_ids'] = np.array(self.midi_data_list[idx]['src_ids'])
            out_dict['src_mask'] = np.array(self.midi_data_list[idx]['src_mask'])
        return arr, out_dict

# 分類器あり (CGモデル)
# データ関連の定義
class LargeMidiDataset(Dataset):
    def __init__(
            self, padded_tokens_list, embedding_model, data_args, eigen_transform=None,
            mapping_func=None, model_emb=None
    ):
        super().__init__()
        self.padded_tokens_list = padded_tokens_list
        self.embedding_model = embedding_model
        self.length = len(self.padded_tokens_list)
        self.data_args = data_args
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        padded_tokens = self.padded_tokens_list[idx]
        arr = np.array(self.embedding_model(torch.tensor(padded_tokens)).cpu().tolist(), dtype=np.float32)
        if self.eigen_transform is not None:
            old_shape = arr.shape
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)

        if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
            arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

        out_dict = {'input_ids': np.array(padded_tokens)}
        if self.data_args.experiment_mode == 'conditional_gen':  # TODO not implementing conditional gen for now
            raise NotImplementedError
        return arr, out_dict

# 分類器なし (CFGモデル)
# データ関連の定義
class LargeMidiDataset_CFG(Dataset):
    def __init__(
            self, padded_tokens_list, embedding_model, data_args, eigen_transform=None,
            mapping_func=None, model_emb=None, trainfine='base'
    ):
        super().__init__()
        self.padded_tokens_list = padded_tokens_list
        self.embedding_model = embedding_model
        self.length = len(self.padded_tokens_list) # len(self.text_datasets['train'])
        self.data_args = data_args
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        self.trainfine = data_args.trainfine

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.trainfine == 'base': # (base)
            padded_tokens = self.padded_tokens_list[idx]
        else : # (finetune)
            padded_tokens = self.padded_tokens_list[idx]
            print(padded_tokens.shape)
            
        arr = np.array(self.embedding_model(torch.tensor(padded_tokens[:256], dtype=torch.long)).cpu().tolist(), dtype=np.float32)
        print(arr.shape)   
        if self.eigen_transform is not None:
            old_shape = arr.shape
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)
        if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
            arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
            print(arr.shape)
        
        # データ格納定義：MIDIトークン & 条件のデータ
        out_dict = {}
        if self.trainfine == 'base': # 事前学習の場合
            out_dict['input_ids'] = np.array(padded_tokens) #['input_ids'])
            out_dict['emotion'] = np.zeros(100) #np.array(self.padded_tokens_list[idx]['emotion'])
            print('no emotion dataset')
        else : # 再学習の場合
            out_dict['input_ids'] = np.array(padded_tokens[:256]) #self.padded_tokens_list[idx]['input_ids'])
            out_dict['emotion'] = np.array(padded_tokens[256:]) # [1,100] 
        
        if self.data_args.experiment_mode == 'conditional_gen':  # TODO not implementing conditional gen for now
            raise NotImplementedError
            
        return arr, out_dict
    

    
# トークン化
def __tokenize(data_args, split, dataset_partition, tokenizer):
    # data_args.data_path
    try:
        tokenizer.vocab['SOS_None']
        has_sos_eos = True
    except KeyError:
        has_sos_eos = False
    tokens_list = []
    print(f"Start tokenize files in {os.path.join(data_args.data_path, split)} with partition={dataset_partition}")
    for midi_file_name in os.listdir(os.path.join(data_args.data_path, split)):
        if random.random() > dataset_partition:
            continue
        if midi_file_name.endswith('.mid'):
            # will have a very long size for each
            tokens = tokenizer.midi_to_tokens(MidiFile(os.path.join(data_args.data_path, split, midi_file_name)))
            try:
                if has_sos_eos:
                    tokens_list.append([tokenizer.vocab['SOS_None']] + tokens[0] + [tokenizer.vocab['EOS_None']])
                else:
                    tokens_list.append(tokens[0])
            except Exception as e:
                print(f'error on {midi_file_name}')
                print(e)
    print(f'Finish tokenize {len(tokens_list)} items')
    return tokens_list

# 256長のトークンを作成
def __generate_input_ids(tokenizer, data_args, split, dataset_partition, to_save_token_list_path):
    tokens_list = __tokenize(data_args, split, dataset_partition, tokenizer)
    print(f"Start padding...")
    padded_tokens_list = __padding(data_args, tokens_list, data_args.image_size ** 2)
    print(f"Save padded data...")
    np.savez(to_save_token_list_path, padded_tokens_list)
    return padded_tokens_list

# データを順番に格納
def __generate_data_list(padded_tokens_list, embedding_model):
    print('Start hidden state embedding...')
    data_list = [
        {
            'input_ids': padded_tokens,
            'hidden_states': embedding_model(torch.tensor(padded_tokens)).cpu().tolist()
        }
        for padded_tokens in padded_tokens_list
    ]
    return data_list


# データセット作成　& 読み込み

# 分類器あり (CGモデル)
# 一からトークンデータを作成
def create_midi_dataloader(
        *, batch_size, data_args=None, split='train', embedding_model=None, dataset_partition=1
):
    point_debug(data_args)
    print("Creating midi dataloader...")
    to_save_token_list_path = f'{data_args.checkpoint_path}/padded_tokens_list_{split}.npz'
    padded_tokens_list = None
    if data_args.reuse_tokenized_data:
        print('reusing tokenized data...')
        try:
            padded_tokens_list = np.load(to_save_token_list_path)['arr_0']
            print(f'Pre-padded token list loaded from {to_save_token_list_path}.')
        except FileNotFoundError:
            pass
    tokenizer = get_tokenizer(data_args)
    if padded_tokens_list is None:
        padded_tokens_list = __generate_input_ids(
            tokenizer, data_args, split, dataset_partition, to_save_token_list_path
        )
    if not embedding_model:
        print('****** create new embedding model ******')
        embedding_model = __create_embedding_model(data_args, vocab_size=len(tokenizer.vocab))

    use_large_dataset = True
    if use_large_dataset:
        print('Making large Dataset...')
        print()
        dataset = LargeMidiDataset(padded_tokens_list, embedding_model, data_args)
        
    else:
        print('Making Dataset...')
        dataset = MidiDataset(
            __generate_data_list(padded_tokens_list, embedding_model),
            data_args.image_size,
            data_args,
            model_arch=data_args.model_arch,  # transformer for NLP / MIDI, or probably use better music transformer? TODO
        )
    print('Making DataLoader...')
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 64,
        drop_last=True,
        shuffle=True,
        num_workers=1,
    )
    print('Finish making DataLoader...')
    while True:
        yield from data_loader
    

# 分類器なし (CFGモデル)
# (事前学習 & 再学習)
# 予め作成されたデータセットを使用
def create_midi_dataloader_cfg(
        *, batch_size, data_args=None, split='train', embedding_model=None, dataset_partition=1
):
    point_debug(data_args)
    print("Creating midi dataloader CFG")
    padded_tokens_list = None
    # データ読み込み
    if data_args.trainfine == 'base': # (base)
       to_save_token_list_path = f'{data_args.checkpoint_path}/padded_tokens_list_{split}.npz' 
       print("Save midi dataloader CFG base")
    else: # (finetune)
        print('reusing tokenized data...')
        padded_tokens_list_token = None
        padded_tokens_list_attribute = None
        to_save_token_list_path =  f'{data_args.data_path}/{split}_tokens_sort_CFG.npz'
        #★ to_save_attribute_list_path =  f'{data_args.data_path}/{split}_attribute_100.npz' # 音楽属性値
        to_save_attribute_list_path =  f'{data_args.data_path}/attribute100_threshold_{split}_CFG.npz' # 音楽属性値　-　閾値
        padded_tokens_list_token = np.nan_to_num(np.load(to_save_token_list_path)['arr_0'])
        padded_tokens_list_attribute  = np.load(to_save_attribute_list_path)['arr_0']
        print('token ='+str(padded_tokens_list_token.shape))
        print('attribute ='+str(padded_tokens_list_attribute.shape))
        # 1. 音楽属性値を変える：100→36
        mask18_2 = np.zeros_like(padded_tokens_list_attribute)
        number_a18_2 = [1, 2, 3, 5, 20, 22, 25, 26, 35, 37, 40, 41, 42, 43, 44, 45, 49, 57, 62, 63, 64, 67, 68, 69, 70, 71, 73, 75, 77, 81, 84, 85, 87, 92, 95, 99] # 音楽属性値18*2
        mask18_2[:,number_a18_2] = 1.0 #100→36
        # 2. 閾値処理の重みあり・なし
        thresholds = np.load("../improved-diffusion/emogen/data/threshold.npy", allow_pickle=True) #閾値
        thresholds_rep = thresholds.repeat(padded_tokens_list_attribute.shape[0],1).T #閾値の繰り返し
        thresholds_weight0 = torch.load(f'../improved-diffusion/emogen/std_mean_{split}_a18_2.pt') # train/validそれぞれの重み
        thresholds_weight = torch.where(thresholds_weight0[0,:] == 0.0, (torch.tensor(0.0).float()), ((1.0 / thresholds_weight0[0,:]).float())) 
        thresholds_weight_rep = thresholds_weight.unsqueeze(dim=1).T.repeat(padded_tokens_list_attribute.shape[0],1).numpy() #重みの繰り返し
        # 1(マスク)、2(バランス整える重み)をattributeにかける
        padded_tokens_list_attribute18_2 = np.nan_to_num((padded_tokens_list_attribute - thresholds_rep) * mask18_2 ) * thresholds_weight_rep 
        print(padded_tokens_list_attribute18_2[0])
        padded_tokens_list = np.concatenate([padded_tokens_list_token,padded_tokens_list_attribute18_2],1) # 1. 音楽属性値18*2
        print(padded_tokens_list.shape)
        print(f'Pre-padded CFG token list loaded from {to_save_token_list_path}.')
        
        
    # トークン化
    tokenizer = get_tokenizer(data_args)
    if padded_tokens_list is None: # (baseのみ)
        padded_tokens_list = __generate_input_ids(
            tokenizer, data_args, split, dataset_partition, to_save_token_list_path
        )
    # 埋め込みモデルの定義　(データセット作成に必要)
    if not embedding_model:
        print('****** create new embedding model ******')
        print(len(tokenizer.vocab))
        embedding_model = __create_embedding_model(data_args, vocab_size=len(tokenizer.vocab))

    use_large_dataset = True
    if use_large_dataset: # こっちを使用
        print('Making large Dataset...') 
        dataset = LargeMidiDataset_CFG(padded_tokens_list, embedding_model, data_args)
    else:
        print('Making Dataset...') 
        dataset = MidiDataset(
            __generate_data_list(padded_tokens_list, embedding_model),
            data_args.image_size,
            data_args,
            model_arch=data_args.model_arch,  # transformer for NLP / MIDI, or probably use better music transformer? TODO
        )
    print(type(dataset))    
    print('Making DataLoader...')
    data_loader = DataLoader(
        dataset, # 
        batch_size=batch_size,  # 64,
        #sampler=sampler,
        drop_last=True,
        shuffle=True,
        num_workers=1,
    )
    print('Finish making DataLoader...')
    while True:
        yield from data_loader

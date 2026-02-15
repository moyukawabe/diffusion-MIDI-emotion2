"""
# (main)
# 音楽生成!!! 
# (分類器なし・CFGモデル)
"""

import os, json, sys
import torch as th
import numpy as np
import pickle
from sklearn.linear_model import Ridge , LinearRegression
import torch.nn.functional as F

from music_classifier.simplified_transformer_net import SimplifiedTransformerNetClassifierModel
from music_classifier.transfomer_net import TransformerNetClassifierModel
from symbolic_music.rounding import tokens_list_to_midi_list
from symbolic_music.utils import get_tokenizer
from transformers import set_seed, BertConfig
import torch.distributed as dist
from improved_diffusion.test_util import denoised_fn_round
from functools import partial
from improved_diffusion import logger , dist_util_sqs, logger # dist_util
from symbolic_music.scripts.infill_util import langevin_fn3, prepare_args, create_model, create_model_CFG, create_embedding, save_results

from emogen.jSymbolic_classifier_nothres_regression import JSymbolicNet 


def main():
    # モデルの定義やや初期設定など
    set_seed(101)
    args = prepare_args() # CFG trainfine='generation' 
    dist_util_sqs.setup_dist()   # dist_util.setup_dist() # DEBUG ** 
    logger.configure()
    # モデル
    model, diffusion = create_model_CFG(args)  
    frozen_embedding_model = create_embedding(args, model)
    # トークン化の定義
    tokenizer = get_tokenizer(args)  # TODO
    logger.log('load the partial sequences')
    pad_token = tokenizer.vocab['PAD_None']
    # length = 64
    right_pad = th.empty(args.image_size ** 2).fill_(pad_token).long()  #空のトークン
    encoded_partial_seqs = [th.cat([right_pad], dim=0)]
    
    


# EMOTIONマッピング読み込み 
    command_input = th.load('./input_emotion_*/input_emotion_18_2_DEAM.pt')[2,:].unsqueeze(0) 
    print(command_input)

    # command_inputに対する処理
    # 1. 18*2要素以外マスク (18*2の場合)
    thresholds = np.load('/home/moyu/program/Diffusion-LM-on-Symbolic-Music-Generation/improved-diffusion/emogen/data/threshold.npy', allow_pickle=True)
    mask18_2 = np.zeros_like(thresholds)
    number_a18_2 = [1, 2, 3, 5, 20, 22, 25, 26, 35, 37, 40, 41, 42, 43, 44, 45, 49, 57, 62, 63, 64, 67, 68, 69, 70, 71, 73, 75, 77, 81, 84, 85, 87, 92, 95, 99] # 音楽属性値18*2
    mask18_2[number_a18_2] = 1.0 #100→36
    thresholds_rep= (th.from_numpy(thresholds * mask18_2).T) 
    # 2. データの分散 (thresholds_weight0[0,:]) を使用して、重みづけ
    thresholds_weight0 = th.load('./std_mean_train_a18_2.pt') #  100つ：thresholds_weight0 = th.load('./emogen/std_mean.pt')
    thresholds_weight = th.where(thresholds_weight0[0,:] == 0.0, (th.tensor(0.0).float()), ((1.0 / thresholds_weight0[0,:]).float())) 
    command_input_one_zero0 =  (command_input - thresholds_rep) * thresholds_weight
    # 3. 座標値と、trainデータの大きさを揃える
    weight_same_traindata = np.load('./DEAM_train_weight.npz')['arr_0']     # DEAM
    input_same_traindata =  command_input_one_zero0 * weight_same_traindata#.cuda()
    command_input_one_zero = input_same_traindata.repeat(int(args.batch_size/2),1)  

    
    
    # 準備
    control_constraints = []
    debug_lst = []
    # CFG +++
    if args.trainfine is None: #CFGのみ！
        langevin_fn_selected = partial(
                    langevin_fn3, debug_lst, args, model.cuda() if th.cuda.is_available() else model, diffusion,
                    frozen_embedding_model.cuda() if th.cuda.is_available() else frozen_embedding_model,
                    command_input_one_zero, #6/27
                    #command_input.expand(args.batch_size).cuda() if th.cuda.is_available() else command_input.expand(args.batch_size),
                    0.1
                )
    else:
        langevin_fn_selected = None
    control_constraints.append((langevin_fn_selected, command_input_one_zero))
    encoded_partial_seqs = [encoded_partial_seqs[0] for _ in range(len(control_constraints))]
    assert len(control_constraints) == len(encoded_partial_seqs)
    print(f'RUNNING FOR {len(control_constraints)} constraints.', '*-' * 20)
    print(encoded_partial_seqs[0], len(encoded_partial_seqs[0]))


    # 生成
    logger.log("sampling...")
    # CFG +++++++
    sample_dict = {}
    for (encoded_seq, (langevin_fn_selected, command_input_one_zero)) in zip(encoded_partial_seqs, control_constraints):
    # CFG (終) ++++
        all_images = []
        print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            print(encoded_seq.shape)
            # seq expanded to batch
            encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size, -1) #!
            print(frozen_embedding_model.weight.device, encoded_seq.device)
            encoded_seq_hidden = frozen_embedding_model(encoded_seq.cuda() if th.cuda.is_available() else encoded_seq)
            seqlen = encoded_seq.size(1) #!
            sample_shape = (args.batch_size, seqlen, args.in_channel,) #!
            # CFG ++++
            emotion_val = th.cat((command_input_one_zero, th.zeros_like(command_input_one_zero)), dim=0) # emotion_valはバッチサイズ二倍（感情＋０のテンソル）
            # 条件なし生成 の場合
            # emotion_val = th.cat((th.zeros_like(command_input_one_zero), th.zeros_like(command_input_one_zero)), dim=0) # emotion_valはバッチサイズ二倍（感情＋０のテンソル）
            model_kwargs = {'emotion': emotion_val} 
            # CFG (終) ++++
            
            # ノイズ除去
            if args.eval_task_.startswith('control'):
                print('-*' * 200, command_input_one_zero[0], '-*' * 200)
                if args.use_ddim: # こっち
                    loop_func_ = diffusion.ddim_sample_loop_progressive
                else:
                    loop_func_ = diffusion.p_sample_loop_progressive
                for sample in loop_func_(
                        model,
                        args, # CFG
                        sample_shape,
                        denoised_fn=partial(denoised_fn_round, args, frozen_embedding_model.cuda() if th.cuda.is_available() else frozen_embedding_model),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        device=encoded_seq_hidden.device,
                        langevin_fn=langevin_fn_selected,
                        eta=args.eta,
                ):
                    final = sample["sample"]      

            sample = final
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        # CFG：条件なし部分を棄却 ++++
        arr = np.concatenate(all_images, axis=0)
        arr2 = arr[:int(args.batch_size / 2)]
        arr = arr[: args.num_samples]
        start_count = int(len(sample_dict)*args.batch_size/2)
        end_count = int(args.batch_size/2)+start_count
        end_count = list(range(start_count, end_count))
        print(end_count)
        sample_dict[tuple(end_count)] = arr2
                # get length of sample_dict
        print(f"len(sample_dict): {len(sample_dict)}")
        print(f'writing to sample_dict, for {len(sample_dict)*args.batch_size/2}samples')
        # CFG(終) ++++
         
        dist.barrier()
        logger.log("sampling complete")

        print('decoding for e2e', )
        for v in sample_dict.values():
            x_t = v
        print(x_t)
        reshaped = th.tensor(x_t).to(dist_util_sqs.dev())
        print(sample_dict.items())
        logits = model.get_logits(reshaped)
        cands = th.topk(logits, k=1, dim=-1)
        np.save('./input_emotion_miditok2', cands.indices.detach().cpu().numpy()) # midiトークン保存
        save_results(args, sample, tokens_list_to_midi_list(args, cands.indices),None)
    return args


if __name__ == "__main__":
    main_args = main()
    


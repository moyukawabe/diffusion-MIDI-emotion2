"""
# (main)
# 音楽生成!!! 
# (分類器あり・CGモデル)
"""

import os, json, sys
import torch as th
import numpy as np
import pickle
from sklearn.linear_model import Ridge , LinearRegression

from music_classifier.simplified_transformer_net import SimplifiedTransformerNetClassifierModel
from music_classifier.transfomer_net import TransformerNetClassifierModel
from symbolic_music.rounding import tokens_list_to_midi_list
from symbolic_music.utils import get_tokenizer
from transformers import set_seed, BertConfig
import torch.distributed as dist
from improved_diffusion.test_util import denoised_fn_round
from functools import partial
from improved_diffusion import logger
from symbolic_music.scripts.infill_util import langevin_fn3, prepare_args, create_model, create_embedding, save_results

from emogen.jSymbolic_classifier_nothres_regression import JSymbolicNet 


def main():
    # モデルの定義や初期設定など
    set_seed(101)
    args = prepare_args()
    # モデル
    model, diffusion = create_model(args)  
    frozen_embedding_model = create_embedding(args, model)
    # トークン化の定義
    tokenizer = get_tokenizer(args)  # TODO
    logger.log('load the partial sequences')
    pad_token = tokenizer.vocab['PAD_None']
    right_pad = th.empty(args.image_size ** 2).fill_(pad_token).long()  #空のトークン
    encoded_partial_seqs = [th.cat([right_pad], dim=0)]
    

# [emotion]読み込み　
    # 入力：座標値(100つの音楽属性)
    '''
    command_list = np.load('.npz')  
    command_input = []
    command_input.append(command_list['arr_0']) # Q1の場合
    '''
    # 入力：座標値(18*2つの音楽属性)
    # ：細かい変更 > infill_util.py
    command_input = th.load('.pt')[0,:].unsqueeze(0) 
    print(command_input)  

    
# MIDI → 音楽属性値の線形回帰モデル読み込み
    logger.log('load the MIDI_music_attribute_Linear')
    newmodel = JSymbolicNet()
    newmodel.load_state_dict(th.load('./classifier/traindatamidi_musicat_Linear.pth', map_location="cuda")) 
    args.music_attribute = newmodel
    print(args.music_attribute)


# 準備
    control_constraints = []
    debug_lst = []
    langevin_fn_selected = partial(
                langevin_fn3, debug_lst, args, model.cuda() if th.cuda.is_available() else model, diffusion,
                frozen_embedding_model.cuda() if th.cuda.is_available() else frozen_embedding_model,
                command_input,
                #command_input.expand(args.batch_size).cuda() if th.cuda.is_available() else command_input.expand(args.batch_size),
                0.1
            )
    control_constraints.append((langevin_fn_selected, command_input)) # 生成制御
    encoded_partial_seqs = [encoded_partial_seqs[0] for _ in range(len(control_constraints))]
    assert len(control_constraints) == len(encoded_partial_seqs)
    print(f'RUNNING FOR {len(control_constraints)} constraints.', '*-' * 20)

    print(encoded_partial_seqs[0], len(encoded_partial_seqs[0]))


# 生成
    logger.log("sampling...")
    for (encoded_seq, (langevin_fn_selected, command_input)) in zip(encoded_partial_seqs, control_constraints):
        all_images = []
        print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            print(encoded_seq.shape)
            encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size, -1) # seq expanded to batch
            print(frozen_embedding_model.weight.device, encoded_seq.device)
            encoded_seq_hidden = frozen_embedding_model(encoded_seq.cuda() if th.cuda.is_available() else encoded_seq)
            seqlen = encoded_seq.size(1)
            sample_shape = (args.batch_size, seqlen, args.in_channel,)

            # ノイズ除去
            if args.eval_task_.startswith('control'):
                print('-*' * 200, command_input, '-*' * 200)
                if args.use_ddim: # こっち
                    loop_func_ = diffusion.ddim_sample_loop_progressive
                else:
                    loop_func_ = diffusion.p_sample_loop_progressive

                for sample in loop_func_(
                        model,
                        args, 
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

        dist.barrier()
        logger.log("sampling complete")

        print('decoding for e2e', )
        print(sample.shape)
        x_t = sample
        logits = model.get_logits(x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        np.save('./input_emotion_miditok2', cands.indices.detach().cpu().numpy())
        save_results(args, sample, tokens_list_to_midi_list(args, cands.indices),None)
    return args



if __name__ == "__main__":
    main_args = main()

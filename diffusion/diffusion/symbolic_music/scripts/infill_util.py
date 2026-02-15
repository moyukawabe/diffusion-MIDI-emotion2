'''
・langevin_fn3 : 生成の分類器処理 (CGモデル)
・変数の初期設定、引数
・モデルの読み込み
・データ保存
    等に関連するコード
'''

import argparse
import os, json

import numpy as np
import torch as th

from symbolic_music.rounding import load_embedding_model
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.test_util import get_weights
from improved_diffusion import dist_util_sqs, logger #dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_model_and_diffusion_CFG,
    create_model_and_diffusion_CFG2,
    args_to_dict, add_dict_to_argparser,
)
#emogen
from emogen.jSymbolic_classifier_nothres import JSymbolic_classifier_nothres
from emogen.jSymbolic_classifier_nothres_regression import JSymbolic_classifier_nothres_regression  # tanh(活性化関数) 05/01
from symbolic_music.rounding import tokens_list_to_midi_list
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from improved_diffusion.gaussian_diffusion import GaussianDiffusion

j = 0


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, batch_size=1, model_path="",
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0,
        control_model_type='normal',
        music_attribute_regression="emogen/traindatamidi_musicat_Linear.pth", # 音楽属性値回帰モデル　
        trainfine='generation' # CFG
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def prepare_args():
    set_seed(101)
    args = create_argparser().parse_args()
    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    dist_util_sqs.setup_dist()  # dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    return args

# CGモデルの場合
def create_model(args):
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    path = dist_util_sqs.load_state_dict(args.model_path, map_location="cpu") # dist_util
    restored_path = {}
    for k,v in path.items():
        if k.startswith('module.'):
            k = k[7:]
        restored_path[k] = v
    model.load_state_dict(restored_path)
    model.to(dist_util_sqs.dev()) 
    model.eval()
    return model, diffusion

# CFGモデルの場合
def create_model_CFG(args):
    logger.log("creating model and diffusion CFG...")
    model, diffusion = create_model_and_diffusion_CFG2( #2
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    path = dist_util_sqs.load_state_dict(args.model_path, map_location="cpu")
    restored_path = {}
    if args.trainfine==None:
        for k,v in path.items():
            if k.startswith('module.'):
                k = k[7:]
            restored_path[k] = v
    else:
        restored_path = path
    model.load_state_dict(restored_path)
    # model.to(dist_util_sqs.dev()) 
    model.eval()
    return model, diffusion

# 学習済みモデルの読み込み
def create_embedding(args, model):
    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    model_embs = load_embedding_model(args)
    model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda() if th.cuda.is_available() else model_embs
    return get_weights(model_embs, args)

# 生成音楽を保存
def save_results(args, samples, midi_list, extra_id=None):
    # sample saving
    try:
        samples = samples.cpu()
        if dist.get_rank() == 0:
            model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
            out_path = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, samples)
        dist.barrier()
        if args.verbose == 'yes':
            # create midi files
            for i, midi in enumerate(midi_list):
                out_path2 = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}_{extra_id or ''}_{i}.mid")
                midi.dump(out_path2)
    except Exception as e:
        import pdb
        pdb.set_trace()

# 使わない
def get_score(input_embs, label_ids, model_control, t=None):
    label_ids2 = label_ids.clone()
    label_ids2[:, :65] = -100
    # print(label_ids2[:, 65:])
    # print(final.shape, tgt_embs.shape)
    # input_embs = th.cat([final, tgt_embs], dim=1)
    model_out = model_control(input_embs=input_embs,
                              labels=label_ids2, t=t)
    print(model_out.loss, 'final end')
    loss_fn = th.nn.CrossEntropyLoss(reduction='none')
    shifted_logits = model_out.logits[:, :-1].contiguous()
    shifted_labels = label_ids2[:, 1:].contiguous()
    loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)).reshape(
        shifted_labels.shape)
    return loss.sum(dim=-1).tolist()


# 生成：CGモデルの分類器処理
# コメントアウト部分は、回帰モデルを使用しない場合(うまくいかなかった)
def langevin_fn3(debug_lst, args, model_control, diffusion, frozen_embedding_model, labels, step_size, sample, mean, sigma,
                 alpha, t, prev_sample):  # current best.
    if t[0].item() < 10: # 最後の10ステップは繰り返し1回
        K = 1
    else:
        K = 3
    if t[0].item() > 0:
        tt = t[0].item() - 1
    else:
        tt = 200
    
    input_embs_param = th.nn.Parameter(sample)
    
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size ,initial_accumulator_value=0.1 ) 
            print(optimizer.param_groups)
            optimizer.zero_grad()

            # 分類器
            jsmbolic_c = JSymbolic_classifier_nothres_regression() 
            # MIDIトークン化(丸め込み)
            get_logits = model_control.get_logits    
            logits  = diffusion.jsymbolic_pre(input_embs_param, get_logits) 
                
            '''''    
            # logits → cands : MIDI → MIDIデータに変換
            # 正確な音楽属性値を取得(jsymbolic)する場合
            # logits_xtg_mid = tokens_list_to_midi_list(args, cands_xt.indices)
            input_embs_param_data = th.tensor(input_embs_param.data.clone(), requires_grad=True).retain_grad() 
            cands_xtgen  = diffusion.jsymbolic_pre(input_embs_param_data, get_logits) # [16,100(256),32]
            cands_xtgen_indices = th.tensor(cands_xtgen.clone(),requires_grad=True)  #.to(th.float32)
            cands_xtgen_indices.retain_grad()
            model_output_jc = jsmbolic_c.main(cands_xtgen_indices.squeeze(), args) #0627
            ''''' 
            model_output_jc  = jsmbolic_c.main(logits, args) 
            model_output_jc.retain_grad()

            coef = 0.01
            loss_fct = MSELoss()
            m = th.nn.Sigmoid()
            print(model_output_jc.shape) #[1,16,100]
            input_embs_param_clone = input_embs_param.clone()
            
            # 損失計算
            L = 7 # 勾配更新の合計回数
            rep_i = 0 # 初期化
            js_loss = th.zeros((1, 7),requires_grad=True).cuda() # 初期化  
            js_loss.retain_grad()
            # 複数損失計算 : for文7回繰り返す        
            for rep_i in range(L): # 複数損失計算　→ L:勾配更新の番号 (0~6)                  
                # マスク
                mask = th.zeros((args.batch_size, 100 ,args.in_channel),requires_grad=False).to(th.float64).cuda() # サイズが？
                print(mask.dtype)
                # rep_i：勾配更新繰り返し回数 = 音楽要素の順番
                # ピッチ関連
                if rep_i == 0: 
                    # mask[:,0:16,:] = 1 
                    mask[:,1:4,:] = 1 # ピッチを厳選
                    mask[:,5,:] = 1 # ピッチを厳選
                    elements = 4.0 # 要素数 # elements = 16.0
                # メロディー関連
                elif rep_i == 1: 
                    # mask[:,16:30,:] = 1 
                    mask[:,20,:] = 1 # メロディーを厳選 
                    mask[:,22,:] = 1 # メロディーを厳選 
                    mask[:,25,:] = 1 # メロディーを厳選 
                    mask[:,26,:] = 1 # メロディーを厳選 
                    elements = 4.0 # elements = 14.0
                # コード関連
                elif rep_i == 2: 
                    # mask[:,30:58,:] = 1 
                    mask[:,35,:] = 1 # コードを厳選
                    mask[:,37,:] = 1  # コードを厳選
                    mask[:,40:46,:] = 1  # コードを厳選
                    mask[:,49,:] = 1  # コードを厳選
                    mask[:,57,:] = 1  # コードを厳選
                    elements = 10.0 # elements = 28.0
                # リズム関連
                elif rep_i == 3: 
                    # mask[:,58:94,:] = 1 
                    mask[:,62:65,:] = 1 # リズムを厳選
                    mask[:,67:72,:] = 1 # リズムを厳選
                    mask[:,73,:] = 1 # リズムを厳選
                    mask[:,75,:] = 1 # リズムを厳選
                    mask[:,77,:] = 1 # リズムを厳選
                    mask[:,81,:] = 1 # リズムを厳選
                    mask[:,84:86,:] = 1 # リズムを厳選
                    mask[:,87,:] = 1 # リズムを厳選
                    mask[:,92,:] = 1 # リズムを厳選
                    elements = 16.0 # elements = 36.0
                # リズム (テンポ・ダイナミクス) 関連
                elif rep_i == 4:
                    # mask[:,94:96,:] = 1 
                    mask[:,95,:] = 1 # リズム (テンポ・ダイナミクス)厳選 
                    elements = 1.0 # elements = 2.0
                # テクスチャー関連
                elif rep_i == 5:
                    # mask[:,96:98,:] = 1 # テクスチャー厳選 : なし
                    elements = 1.0 # elements = 2.0
                # ダイナミクス関連
                else:
                    # mask[:,98:100,:] = 1 
                    mask[:,99,:] = 1 # ダイナミクス厳選
                    elements = 1.0 # elements = 2.0
                    
                target_jc = ((th.Tensor(labels).float()).cuda().repeat((args.num_samples,1))) # [64,100,218]
                
                
            # 閾値処理  (Russellデータとtrainデータの重みを揃える)++++++++++
                # 音楽属性の実値から閾値をひき、データの分散 (thresholds_weight0[0,:]) を使用して重みづけ
                # 閾値
                thresholds = np.load('./threshold.npy', allow_pickle=True)
                thresholds_rep= (th.from_numpy(thresholds.T)).repeat(args.num_samples ,1)
                # 閾値にかける重み
                # thresholds_weight0 = th.load('./std_mean.pt').cuda() # ① 100音楽属性の場合 
                thresholds_weight0 = th.load('./std_mean_train_a18_2.pt').cuda() # ② 18*2音楽属性の場合
                thresholds_weight = th.where(thresholds_weight0[0,:] == 0.0, (th.tensor(0.0).float()).cuda(), ((1.0 / thresholds_weight0[0,:]).float()).cuda()) 
                thresholds_weight = th.unsqueeze(thresholds_weight,dim=0).repeat(args.num_samples,1) 
                one_zero =  (target_jc.cuda() - thresholds_rep.cuda()) * thresholds_weight.cuda() # 重み：jSymbolic_classifier_nothres_regression.py
                # Russellデータと、trainデータの大きさを揃える
                weight_same_traindata = np.load('./DEAM_train_weight.npz')['arr_0']
                weight_same_traindata = th.from_numpy(weight_same_traindata.astype(np.float32)).clone()
                input_same_traindata = one_zero * weight_same_traindata.cuda()
                target_jc_mask = input_same_traindata * mask[:,:,0] 
            # 閾値処理 +++++++++++
        
        
            # ◎予測データ (model_output_jc )
                model_output_jc_mask = model_output_jc * mask
                print('model_output_jc_mask =' + str(model_output_jc_mask))
                  
            
            # classifierの損失の合計
                js_loss_0 = []
                ii = 0
                weight_js = 1000.0 # 属性値比較の重み
                for ii in range(args.in_channel):
                    js_loss_0.append(loss_fct(model_output_jc_mask[:,:,ii], target_jc_mask).mean(dim=-1).sum()) # 10/18
                    ii = ii + 1
                js_loss_0  = th.stack(js_loss_0, dim = 0).mean(dim=-1).sum() / elements # 属性値の要素数elementsで割って、lossの平均を求める (10/10)
                js_loss[0,rep_i]  = js_loss_0
                # ★ js_loss16 = loss_fct((th.Tensor(model_output_jc).float()).reshape(i,100), (th.Tensor(label).float()).reshape(i,100)).mean(dim=-1)
            # (+++end) classifierの損失の合計
            
                ''''' 
                ## 勾配調整 +++ (正しい音楽属性値を使用する場合：同じ音楽出力への対策)
                model_output_jc = th.Tensor(model_output_jc).float().reshape(i0,100).cuda()
                ii = 0
                js_loss = th.zeros(1,i0).cuda()
                js_loss16 = 0.0
                while(ii < i0):
                js_loss16 = (js_loss16 + ((model_output_jc[ii] - th.Tensor(np.array(labels)).float().cuda())** 2 / 1.).mean(dim=0).sum()).cuda()
                ## ★ js_loss = th.abs( (model_output_jc - th.Tensor(np.array(labels)).float()).mean(dim=0).sum().cuda()) * th.mean(input_embs_param_clone**2)#.sum()#.cuda()
                ii = ii + 1
                js_loss16 = js_loss16/i0
                # 0626 [1,100]*[16, 256, 218]
                ## 勾配調整 +++ 
                '''''
                print('classifier loss'+ str(rep_i)+ ' = ' + str(js_loss[0,rep_i]))
            # 複数損失計算　(終)
    
            coef = 0.01
                # coef=1.
            # 拡散モデル側の損失
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
            
            # 分類器側の損失を足す
            '''    
            # 勾配調整 +++ (正しい音楽属性値を使用する場合：同じ音楽出力への対策)
            # lgp = logp_term.detach()
            # loss = (js_loss + lgp)/lgp * logp_term #*100  # 02010 (決)割100→grad/100 , (*100→　02025,02026)
            # js = (js_loss16 / lgp) #0416
            # 勾配調整 +++
            '''
            # lossの計算7つ分
            loss =  logp_term 
            loss = logp_term + (js_loss.sum() * weight_js)
            print('all loss = '+str(loss))
            loss.backward(retain_graph=True)
        
            print('input_embs_param .grad = '+ str(input_embs_param.grad))
            print('logit grad = '+ str(logits.grad)) 
            print('classifier .grad = '+ str(model_output_jc.grad))  # 分類器処理の勾配
               
            '''    
            ## 勾配調整 +++ (正しい音楽属性値を使用する場合：同じ音楽出力への対策)
            # input_embs_param.grad = input_embs_param.grad + js * input_embs_param.grad
            ## 勾配調整 +++
            '''    
            optimizer.step()
                
            epsilon = th.randn_like(input_embs_param.data)
            print('after_param.data = '+ str(input_embs_param.data))
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())
        global j
        j = j + 1
        print(j)

    return input_embs_param.data



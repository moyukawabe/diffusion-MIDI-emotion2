"""
学習時、run_trainから呼び出される
CFGモデル (分類器なし)
事前学習
Train a diffusion model on images.
"""

import argparse
import json, torch, os
import time

from improved_diffusion import dist_util_sqs, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion_CFG,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.utils import point_debug
from symbolic_music.datasets import create_midi_dataloader, create_midi_dataloader_cfg 
from symbolic_music.rounding import load_embedding_model
from symbolic_music.utils import is_midi_task
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models, load_tokenizer
import wandb


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util_sqs.setup_dist() # DEBUG **
    #os.environ["OMP_NUM_THREADS"] = "1" # 複数GPU
    logger.configure()

    # make the diffusion model
    logger.log("creating model and diffusion  CFG...")
    model, diffusion = create_model_and_diffusion_CFG(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'the parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

# WandB
    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "diffusion_CGF_lm_base"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    '''
    if args.experiment_mode == 'conditional_gen':
        assert args.modality in ['e2e']
        assert args.padding_mode == 'pad'
    '''
    logger.log("creating data loader...")
    if is_midi_task(args):
        # do midi staff
        logger.log(f'Load base training data...')
        data = create_midi_dataloader_cfg(
            batch_size=args.batch_size,
            data_args=args,
            dataset_partition=args.dataset_partition,
            embedding_model=None
        )
        logger.log(f'Finish base load training data loader...')
        
        time.sleep(1)
        point_debug(args)
        next(data)
        point_debug(args)
        
        logger.log(f'Load embedding model...')
        embedding_model = load_embedding_model(args)
        
        logger.log(f'Load validation data...')
        data_valid = create_midi_dataloader_cfg(
            batch_size=args.batch_size,
            data_args=args,
            split='valid',
            embedding_model=embedding_model,
            dataset_partition=args.dataset_partition
        )
        logger.log(f'Finish load validation data loader...')
    
    

    logger.log("Start mapping...")
    def get_mapping_func(args, diffusion, data):
        if is_midi_task(args):
            model2 = load_embedding_model(args)
        
        model3 = get_weights(model2, args)  # 设置成不需要grad
        print(model3, model3.weight.requires_grad)
        # loss function，提前赋值进 args & embedding model，然后塞给diffusion
        mapping_func = partial(
            compute_logp, args, model3.cuda() if torch.cuda.is_available() else model3
        )
        diffusion.mapping_func = mapping_func
        return mapping_func

    get_mapping_func(args, diffusion, data)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        args = args,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=101,
        gradient_clipping=-1.0,
        eval_interval=2000,
        checkpoint_path='diff_models',
        dataset_partition=1.0,
        debug=False,
        nproc_per_node=2,
        #cuda=7 ##
        trainfine = 'base'
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress',model_arch='conv-unet',
                         roc_train='diffusion_lm/ROCstory',#'diffusion_lm/ROCstory/ROCstory17.csv',
                         wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
                         e2e_train='e2e_data',
                         yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
                         commonGen_train='diffusion_lm/common-gen/commongen_data',
                         data_path='../datasets/midi/giant_midi_piano',
                         emb_scale_factor=1.0, noise_level=0.0, cache_mode='no', use_bert_tokenizer='no',
                         padding_mode='block',
                         preprocessing_num_workers=1,
                         reuse_tokenized_data=False,
                         midi_tokenizer='REMI')
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

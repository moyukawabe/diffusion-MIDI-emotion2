#  Emotion-controllable Music Generation with a Diffusion Models and Musical Attributes

## Referenced codes
Diffusion-LM on Symbolic Music Generation with Controllability
> Repository: https://github.com/SwordElucidator/Diffusion-LM-on-Symbolic-Music-Generation?tab=readme-ov-file
> Paper: https://cs230.stanford.edu/projects_fall_2022/reports/16.pdf
  
EmoGen
> Repository: https://github.com/microsoft/muzic/tree/main/emogen
> Paper: https://arxiv.org/html/2307.01229v1

EMOPIA_cls
> Repository: https://github.com/SeungHeonDoh/EMOPIA_cls

Essentia models
> Page: https://essentia.upf.edu/models.html
  
Guided Text Generation with Classifier-free Language Diffusion
> Repository: https://github.com/vvhg1/guided-text-generation-with-classifier-free-language-diffusion

### Environment
- Basic
  - Use Diffusion models and so on
  - `python 3.8`
  - `pip install -r requirements.txt`
- Use EMOPIA_cls
  - `python 3.8`
  - `pip install -r requirements_evaluate.txt`
- Use Essentia model
  - `python 3.7` (Mac OS)
  - `pip install -r requirements_musicAV.txt`
- Use jSymbolic system on a GPU:
  - `python 3.7`
  - `pip install -r make_data/jSymbolic_use_file/requirements_jSymbolic_gpu.txt`

## Data
### Datasets
1. Training diffusion models
  
  - GiantMIDI-Piano 
    - Repository: https://github.com/bytedance/GiantMIDI-Piano?tab=readme-ov-file
    - Data: https://drive.google.com/file/d/1lmMu8lPo7rsgCIO0yjceap8TUfM72aFK/view?usp=sharing
<br/>
2. Training inpur regression models
   
  - EMOPIA
    - MIDI data: https://zenodo.org/records/5090631#.YPPo-JMzZz8
    - Audio data: obtaining audio data
      - Data: `make_data/EMOPIA/songs_lists/` <br>
      Use YouTube links
      - System's Repository: https://github.com/ytdl-org/youtube-dl
      - Example code: `make_data/EMOPIA/audio_DL.ipynb`

  - DEAM
     - Data: https://cvml.unige.ch/databases/DEAM/

### Make data
- jSymbolic
  - Extracting Music Attribute Values
  - Download the package: https://sourceforge.net/projects/jmir/files/jSymbolic/jSymbolic%202.2/
  - Code: `make_data/jSymbolic_use_file/jSymbolic_feature.py`

- EMOPIA_cls
  - Predicting Emotion's inference values
  - Code: `evaluate/EMOPIA_use_file/inference_many.py`

- Example of creating pair data of "Music Attribute Value" and "MIDI token"
  - Code: `make_data/Giantmidipiano/midi_attribute.ipynb`


## Usage
### Input
 Make "input's regression models" and "Input data for the diffusion models"
- Use EMOPIA data
 - Example code: ` make_data/EMOPIA/emotion_ar_va.ipynb` or `make_data/EMOPIA/emotion_ar_va2.ipynb`
- Use DEAM data
  - Example code: `make_data/DEAM/DEAM_xy_attribute.ipynb `
    
### Training
`mkdir diffusion_models;` <br>
`cd diffusion/diffusion`
- CG (Classifier-Guidance) model
  - Pretraining:
    - Main code: `diffusion/diffusion/scripts/train_base.py` 
    - Rename: diffusion_models → diffusion_model_CG_base <br>
      `CUDA_VISIBLE_DEVICES=7 nohup python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --save_interval 2000  --lr_anneal_steps 200000 --seed 102 --noise_schedule sqrt --in_channel 32 --modality midi --submit no --padding_mode bar_block --app "--predict_xstart False --training_mode e2e --vocab_size 218 --e2
e_train ../datasets/midi/giant_midi_piano " --notes previous_X_midi --dataset_par
tition 1 --image_size 16 --midi_tokenizer='REMI' --data_path ../datasets/midi/giant_midi_piano > output_train_base_CG.txt &`
      
  - Training:
    - Main code: `diffusion/diffusion/scripts/train_finetune.py`
    - Rename: diffusion_models → diffusion_model_CFG_base
      `CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 --use_env scripts/run_train_finetune.py --model_path diffusion_model_CG_base/diff_midi_giant_midi_piano_REMI_bar_block_rand32_transformer_lr0.0001_0.0_4000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model020000.pt --diff_steps 4000 --model_arch transformer --lr 0.0001 --save_interval 8000 —lr_anneal_steps 020000 --seed 102 --noise_schedule sqrt --in_channel 32 --modality midi --submit no --padding_mode bar_block --app "--predict_xstart True --training_mode e2e --vocab_size 218 --e2e_train ../datasets/midi/giant_midi_piano " --notes xstart_midi --dataset_partition 1 --image_size 16 --midi_tokenizer='REMI' --data_path ../datasets/midi/giant_midi_piano > output_train_CG.log 2>&1`
      
- CFG (Classifier-Free Guidance) model
  - Pretraining
    - Main code: `diffusion/diffusion/scripts/train_base_CFG.py` 
     - Rename: diffusion_models → diffusion_model_CG_re <br>
      `CUDA_VISIBLE_DEVICES=0,1 nohup python scripts/run_train_base_CFG.py --diff_steps 4000 --model_arch transformer --lr 0.0001 --save_interval 4000  --lr_anneal_steps 100000 --seed 102 --noise_schedule sqrt --in_channel 32 --modality midi --submit no --padding_mode bar_block --app "--predict_xstart True --training_mode e2e --vocab_size 218 --e2e_train ../datasets/midi/giant_midi_piano " --notes xstart_midi --dataset_partition 1 --bsz 64 --image_size 16 --midi_tokenizer='REMI' --data_path ../datasets/midi/giant_midi_piano > output_train_base_CFG.log 2>&1`

  - Training
    - Main code: `diffusion/diffusion/scripts/train_finetune_CFG.py`
    - Rename: diffusion_models → diffusion_model_CFG_re <br>
      `CUDA_VISIBLE_DEVICES=0,1 nohup python scripts/run_train_finetune_CFG.py --model_path diffusion_model_CFG_base/diff_midi_giant_midi_piano_REMI_bar_block_rand32_transformer_lr0.0001_0.0_4000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model020000.pt --diff_steps 4000 --model_arch transformer --lr 0.0001 --save_interval 4000  --lr_anneal_steps 100000 --seed 102 --noise_schedule sqrt --in_channel 32 --modality midi --submit no --padding_mode bar_block --app "--predict_xstart True --training_mode e2e --vocab_size 218 --e2e_train ../datasets/midi/giant_midi_piano " --notes xstart_midi --dataset_partition 1 --bsz 64 --image_size 16 --midi_tokenizer='REMI' --data_path ../datasets/midi/giant_midi_piano > output_train_CFG.log 2>&1`
      
### Generation
- CG (Classifier-Guidance) model
  - Main code: `diffusion/diffusion/symbolic_music/scripts/control_attribute_emogen_jsymbolic.py` <br>
    `mkdir generation_outputs_CG;`
     `CUDA_VISIBLE_DEVICES=0 nohup python symbolic_music/scripts/control_attribute_emogen_jsymbolic.py --model_path diffusion_model_CG_re/diff_midi_giant_midi_piano_REMI_bar_block_rand32_transformer_lr0.0001_0.0_4000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model012000.pt --eval_task_ control_attribute --tgt_len 230 --use_ddim True --eta 1. --batch_size 16 --num_samples 16 --out_dir generation_outputs_CG > outgen_CG.txt &`
     
- CFG (Classifier-Free Guidance) model
  - Main code: `diffusion/diffusion/symbolic_music/scripts/cfg_control_attribute_emogen_jsymbolic.py` <br>
    `mkdir generation_outputs_CFG;`
    `CUDA_VISIBLE_DEVICES=0,1 nohup python symbolic_music/scripts/cfg_control_attribute_emogen_jsymbolic.py --model_path diffusion_model__CFG_re/diff_midi_giant_midi_piano_REMI_bar_block_rand32_transformer_lr0.0001_0.0_4000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart_midi/model100000.pt --eval_task_ control_attribute --tgt_len 230 --use_ddim True --eta 1. --batch_size 16 --num_samples 16 --out_dir generation_outputs_CFG > outgen_CFG.txt &`


## Evaluate
Use the files under the `evaluate/`
- Emotion estimates: `Emotion.ipynb`
- Music attribute similarity: `Similarity.ipynb`
- Music attribute analysis: `analysis.ipynb`

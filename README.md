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
- CG (Classifier-Guidance) model
  - Pretraining:
    - Main code:
      `mkdir diffusion_model_CG_base;`
      ``
  - Training:
    - Main code:
      `mkdir diffusion_model_CFG_base;`
      ``
- CFG (Classifier-Free Guidance) model
  - Pretraining
    - Main code:
      `mkdir diffusion_model_CG_re;`
      ``
  - Training
    - Main code:
      `mkdir diffusion_model_CFG_re;`
      ``
      
### Generation
- CG (Classifier-Guidance) model
  - Main code:
    `mkdir generation_outputs_CG`
     ``
     
- CFG (Classifier-Free Guidance) model
  - Main code:
    `mkdir generation_outputs_CG`
    ``


## Evaluate
Use the files under the `evaluate/`
- Emotion estimates: `Emotion.ipynb`
- Music attribute similarity: `Similarity.ipynb`
- Music attribute analysis: `analysis.ipynb`

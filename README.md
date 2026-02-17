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

## Datasets
1. Training diffusion models
  
  - GiantMIDI-Piano
  > Repository: https://github.com/bytedance/GiantMIDI-Piano?tab=readme-ov-file
  > Data: https://drive.google.com/file/d/1lmMu8lPo7rsgCIO0yjceap8TUfM72aFK/view?usp=sharing

2. Training inpur regression models
   
  - EMOPIA
    > MIDI data: https://zenodo.org/records/5090631#.YPPo-JMzZz8
    > Audio data: obtaining audio data
      - Data: make_data/EMOPIA/songs_lists 
      Use YouTube links
      - System's Repository: https://github.com/ytdl-org/youtube-dl
      - Example code: make_data/EMOPIA/audio_DL.ipynb

  - DEAM
     > Data: https://cvml.unige.ch/databases/DEAM/

## Make data
- jSymbolic
  - Extracting Music Attribute Values
  > Download the package: https://sourceforge.net/projects/jmir/files/jSymbolic/jSymbolic%202.2/
  - Code: make_data/jSymbolic_use_file/jSymbolic_feature.py

- EMOPIA_cls
  - Predicting Emotion's inference values
  - Code: evaluate/EMOPIA_use_file/inference_many.py

- Example of creating pair data of "Music Attribute Value" and "MIDI token"
  - Code: make_data/Giantmidipiano/midi_attribute.ipynb

## Training


## Evaluate
Use the files under the evaluate/
- Emotion # Emotion.ipynb
- Music attribute similarity #:Similarity.ipynb
- Music attribute analysis #:analysis.ipynb

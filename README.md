<p align="center" width="100%">
<img src="./pandagpt.png" alt="PandaGPT-4" style="width: 40%; min-width: 300px; display: block; margin: auto;">
</p>

# PandaGPT: Empowering Large Language Models with Visual and Auditory Intelligence

![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Model Weight License](https://img.shields.io/badge/Model_Weight%20License-CC%20By%20NC%204.0-red.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)


This repo contains related resources of PandaGPT.

This repo contains
- The <a href='#weights'>delta weights</a> for the fine-tuned model.
- The <a href='#data'>data</a> used for fine-tuning the model.
- The <a href='#example_usage'>example usage</a> of OpenAlpaca.
- The <a href='#code'>code</a> for fine-tuning the model.

**Usage and License Notices:**

****

<span id='all_catelogue'/>

## Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#environment'>2. Running PandaGPT Demo</a>
    * <a href='#install_environment'>2.1. Environment Installation</a>
    * <a href='#download_imagebind_model'>2.2. Prepare ImageBind Checkpoint</a>
    * <a href='#download_vicuna_model'>2.3. Prepare Vicuna Checkpoint</a>
    * <a href='#download_pandagpt'>2.4. Prepare Delta Weights of PandaGPT</a>
    * <a href='#running_demo'>2.5. Deploying Demo</a>

****

<span id='introduction'/>

### 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>

****

<span id='environment'/>

### 2. Running PandaGPT Demo: <a href='#all_catelogue'>[Back to Top]</a>

<span id='install_environment'/>

#### 2.1. Environment Installation:
To install the required environment, please run
```
pip install -r requirements.txt
```

Then install the Pytorch package with the correct cuda version, for example
```
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch/
```

<span id='download_imagebind_model'/>

#### 2.2. Prepare ImageBind Checkpoint:
You can download the pre-trained ImageBind model using [this link](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth). After downloading, put the downloaded file (imagebind_huge.pth) in [[./pretrained_ckpt/imagebind_ckpt/]](./pretrained_ckpt/imagebind_ckpt/) directory. 

<span id='download_vicuna_model'/>

#### 2.3. Prepare Vicuna Checkpoint:
To prepare the pre-trained Vicuna model, please follow the instructions provided [[here]](./pretrained_ckpt#1-prepare-vicuna-checkpoint).


<span id='download_pandagpt'/>

#### 2.4. Prepare Delta Weights of PandaGPT:

|**Base Language Model**|**Learning Tasks**|**Huggingface Delta Weights Address**|
|:-------------:|:-------------:|:-------------:|
|Vicuna-7B (version 0)|Image Captioning|[openllmplayground/pandagpt_7b_v0_image_captioning_only](https://huggingface.co/openllmplayground/pandagpt_7b_v0_image_captioning_only)|
|Vicuna-7B (version 0)|Visual Instruction|[openllmplayground/pandagpt_7b_v0_visual_instruction_only](https://huggingface.co/openllmplayground/pandagpt_7b_v0_visual_instruction_only)|
|Vicuna-7B (version 0)|Image Captioning + Visual Instruction|[openllmplayground/pandagpt_7b_v0](https://huggingface.co/openllmplayground/pandagpt_7b_v0)|
|Vicuna-13B (version 0)|Image Captioning|[openllmplayground/pandagpt_13b_v0_image_captioning_only](https://huggingface.co/openllmplayground/pandagpt_13b_v0_image_captioning_only)|
|Vicuna-13B (version 0)|Visual Instruction|[openllmplayground/pandagpt_13b_v0_visual_instruction_only](https://huggingface.co/openllmplayground/pandagpt_13b_v0_visual_instruction_only)|
|Vicuna-13B (version 0)|Image Captioning + Visual Instruction|[openllmplayground/pandagpt_13b_v0](https://huggingface.co/openllmplayground/pandagpt_13b_v0/)|

We release the delta weights of PandaGPT trained with different strategies in the table above. After downloading, put the downloaded file (pytorch_model.pt) in the [[./pretrained_ckpt/pandagpt_ckpt/]](./pretrained_ckpt/pandagpt_ckpt/) directory.

<span id='running_demo'/>

#### 2.5. Deploying Demo:
Upon completion of previous steps, you can run the demo as
```bash
cd ./code/
python web_demo.py
```





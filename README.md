<p align="center" width="100%">
<img src="./pandagpt.png" alt="PandaGPT-4" style="width: 40%; min-width: 300px; display: block; margin: auto;">
</p>

# PandaGPT: Empowering Large Language Models with Visual and Auditory Intelligence

![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Model Weight License](https://img.shields.io/badge/Model_Weight%20License-CC%20By%20NC%204.0-red.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)


[[Project Page]()] [[Paper]()] [[Demo]()] [[Data](<a href='#all_catelogue'>[Back to Top]</a>)] [[Model]()]

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
* <a href='#train_pandagpt'>3. Train Your Own PandaGPT</a>
    * <a href='#data_preparation'>3.1. Data Preparation</a>
    * <a href='#training_configurations'>3.2. Training Configurations</a>
    * <a href='#model_training'>3.3. Training PandaGPT</a>
* <a href='#license'>Usage and License Notices</a>
* <a href='#acknowledgments'>Acknowledgments</a>

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

****

<span id='train_pandagpt'/>

### 3. Train Your Own PandaGPT: <a href='#all_catelogue'>[Back to Top]</a>

**Prerequisites:** Before training the model, making sure the environment is properly installed and the checkpoints of ImageBind and Vicuna are downloaded. You can refer to [here](https://github.com/yxuansu/PandaGPT#2-running-pandagpt-demo-back-to-top) for more information.  

<span id='data_preparation'/>

#### 3.1. Data Preparation:

**Declaimer:** To ensure the reproducibility of our results, we have released our training datasets. The datasets must be used for research purpose only. The use of the datasets must comply with the licenses from original sources. These datasets may be taken down when requested by the original authors.

|**Stage**|**Training Task**|**Dataset Address**|
|:-------------:|:-------------:|:-------------:|
|1|Image Captioning|[openllmplayground/PandaGPT4_Stage_1_Data](https://huggingface.co/datasets/openllmplayground/PandaGPT4_Stage_1_Data)|
|2|Visual Instruction|[openllmplayground/PandaGPT4_Stage_2_Data](https://huggingface.co/datasets/openllmplayground/PandaGPT4_Stage_2_Data)|

After downloading, put the downloaded file and unzip them under the [[./data/]](./data/) directory.

> **** The directory should look like:

    .
    └── ./data/ 
        └── /stage_1/  
            ├── pandagpt4_stage_1_data.json
            └── /images/
                ├── GCC_train_002582585.jpg
                ├── GCC_train_002429825.jpg
                └── ...
        └── /stage_2/ 
            ├── pandagpt4_visual_instruction_data.json
            └── /images/
                ├── 000000426538.jpg
                ├── 000000306060.jpg
                └── ...
              

<span id='training_configurations'/>

#### 3.2 Training Configurations:

The table below show the training hyperparameters used in our experiments. The hyperparameters are selected based on the constrain of our computational resources, i.e. 8 x A100 (40G) GPUs.

|**Base Language Model**|**Stage**|**Training Task**|**Epoch Number**|**Batch Size**|**Learning Rate**|**Maximum Length**|
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|7B|1|Image Captioning|2|256|5e-4|128|
|7B|2|Visual Instruction|2|64|5e-4|512|
|13B|1|Image Captioning|2|256|5e-4|128|
|13B|2|Visual Instruction|2|64|5e-4|256|



<span id='model_training'/>



#### 3.3. Training PandaGPT:
 
**[Stage 1 Training]** To train the first stage model, please run the following commands.
```yaml
cd ./code/scripts/
chmod +x train_stage_1.sh
cd ..
./scripts/train_stage_1.sh
```

The key arguments of the training script are as follows:
* `--stage`: The stage number.
* `--data_path`: The data path for the json file `pandagpt4_stage_1_data.json`.
* `--image_root_path`: The root path for the downloaded images.
* `--imagebind_ckpt_path`: The path where saves the ImageBind checkpoint `imagebind_huge.pth`.
* `--vicuna_ckpt_path`: The directory that saves the pre-trained Vicuna checkpoints.
* `--max_tgt_len`: The maximum length of training instances.
* `--save_path`: The directory which saves the trained delta weights. This directory will be automatically created.

The epoch number can be set in the `epochs` argument at [./code/config/openllama_peft.yaml](./code/config/openllama_peft.yaml) file. Also note that the `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps` arguments in [./code/dsconfig/openllama_peft_stage_1.json](./code/dsconfig/openllama_peft_stage_1.json) should be set as 16 and 2 for 7B model, and set as 4 and 8 for 13B model.

**[Stage 2 Training]** To train the second stage model, please run the following commands:
```yaml
cd ./code/scripts/
chmod +x train_stage_2.sh
cd ..
./scripts/train_stage_2.sh
```

The key arguments of the training script are as follows:
* `--stage`: The stage number.
* `--data_path`: The data path for the json file `pandagpt4_stage_1_data.json`.
* `--image_root_path`: The root path for the downloaded images.
* `--imagebind_ckpt_path`: The path where saves the ImageBind checkpoint `imagebind_huge.pth`.
* `--vicuna_ckpt_path`: The directory that saves the pre-trained Vicuna checkpoints.
* `--delta_ckpt_path`: The directory that saves the delta weights obtained from stage 1. It should be `{stage_1_save_path}/pytorch_model.pt`.
* `--max_tgt_len`: The maximum length of training instances.
* `--save_path`: The directory which saves the trained delta weights. This directory will be automatically created.

Note that the `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps` arguments in [./code/dsconfig/openllama_peft_stage_2.json](./code/dsconfig/openllama_peft_stage_2.json) should be set as 2 and 4 for 7B model, and set as 1 and 8 for 13B model.


****

<span id='license'/>

### Usage and License Notices:

PandaGPT is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. The delta weights are also CC BY NC 4.0 (allowing only non-commercial use).


****

<span id='acknowledgments'/>

### Acknowledgments:


This repo benefits from [OpenAlpaca](https://github.com/yxuansu/OpenAlpaca), [ImageBind](https://github.com/facebookresearch/ImageBind), [LLaVA](https://github.com/haotian-liu/LLaVA), and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). Thanks for their wonderful works!



 










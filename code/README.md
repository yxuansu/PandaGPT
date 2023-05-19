
## Instructions on Training PandaGPT:

### 1. Prerequisites:
Before training the model, making sure the environment is properly installed and the checkpoints of ImageBind and Vicuna are downloaded. You can refer to [here](https://github.com/yxuansu/PandaGPT#2-running-pandagpt-demo-back-to-top) for more information.

# Data Preparation:

Download all files from [this link](https://huggingface.co/datasets/openllmplayground/PandaGPT4_Stage_1) and put all required files under [./data/stage_1](./data/stage_1).

# Checkpoint Preparation:

Download Vicuna-7b-v0 checkpoint from [this link](https://huggingface.co/openllmplayground/vicuna_7b_v0) and put all required files under [./pretrained_ckpt/vicuna_ckpt/7b_v0/](./pretrained_ckpt/vicuna_ckpt/7b_v0).


# 如何运行Gradio Demo

## 1. 安装好环境

```bash
pip install -r requirements.txt
```

如果出现了torch, torchaudio, torchvision安装失败的问题，请重新使用如下命令:

```bash
pip install torch==1.13.1+cu116 -f http://download.pytorch.org/whl/torch
pip install torchaudio==0.13.1+cu116 -f http://download.pytorch.org/whl/torchaudio
pip install torchvision==0.14.1+cu116 -f http://download.pytorch.org/whl/torchvision
```

然后，准备好模型文件：
1. imagebind的checkpoint文件，这个可以自动下载，存放在`pretrained_ckpt/imagebind_ckpt`下
2. vicuna_7b的checkpoint文件，从[这个链接](https://huggingface.co/openllmplayground/vicuna_7b_v0)下载，放在`pretrained_ckpt/vicuna_ckpt/7b_v0下`
3. lora训练vicuna的checkpoint文件，从[这个链接](https://huggingface.co/openllmplayground/pandagpt_7b_v0_visual_instruction_only)下载，存放在`ckpt/vicuna_7b_v0_peft/stage_2/pytorch_model.pt`

## 2. 启动Demo

```bash
python web_demo.py
```

需要注意的是，当前的目录下存储有`images`, `thermals`, `audios`, `videos`这四个文件夹，分别存储了对应的demo中的examples

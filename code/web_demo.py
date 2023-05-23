from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import os
import ipdb
import gradio as gr
import mdtex2html
from model.openllama import OpenLLAMAPEFTModel
import torch
import json

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/13b_v0',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the 13b model over ...')

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def re_predict(
    input, 
    image_path, 
    audio_path, 
    video_path, 
    thermal_path, 
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    # drop the latest query and answers and generate again
    q, a = history.pop()
    chatbot.pop()
    return predict(q, image_path, audio_path, video_path, thermal_path, chatbot, max_length, top_p, temperature, history, modality_cache)


def predict(
    input, 
    image_path, 
    audio_path, 
    video_path, 
    thermal_path, 
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    if image_path is None and audio_path is None and video_path is None and thermal_path is None:
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
    else:
        print(f'[!] image path: {image_path}\n[!] audio path: {audio_path}\n[!] video path: {video_path}\n[!] thermal path: {thermal_path}')

    # prepare the prompt
    prompt_text = ''
    for idx, (q, a) in enumerate(history):
        if idx == 0:
            prompt_text += f'{q}\n### Assistant: {a}\n###'
        else:
            prompt_text += f' Human: {q}\n### Assistant: {a}\n###'
    if len(history) == 0:
        prompt_text += f'{input}'
    else:
        prompt_text += f' Human: {input}'

    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })
    chatbot.append((parse_text(input), parse_text(response)))
    history.append((input, response))
    return chatbot, history, modality_cache


def reset_user_input():
    return gr.update(value='')

def reset_dialog():
    return [], []

def reset_state():
    return None, None, None, None, [], [], []


with gr.Blocks(scale=4) as demo:
    gr.HTML("""<h1 align="center">PandaGPT</h1>""")

    with gr.Row(scale=4):
        with gr.Column(scale=1):
            image_path = gr.Image(type="filepath", label="Image", value=None)
        with gr.Column(scale=1):
            audio_path = gr.Audio(type="filepath", label="Audio", value=None)
        with gr.Column(scale=1):
            video_path = gr.Video(type='file', label="Video")
        with gr.Column(scale=1):
            thermal_path = gr.Image(type="filepath", label="Thermal Image", value=None)

    chatbot = gr.Chatbot().style(height=300)
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                with gr.Row(scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
                with gr.Row(scale=1):
                    resubmitBtn = gr.Button("Resubmit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 400, value=256, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.01, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=1.0, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    modality_cache = gr.State([])

    submitBtn.click(
        predict, [
            user_input, 
            image_path, 
            audio_path, 
            video_path, 
            thermal_path, 
            chatbot, 
            max_length, 
            top_p, 
            temperature, 
            history, 
            modality_cache,
        ], [
            chatbot, 
            history,
            modality_cache
        ],
        show_progress=True
    )

    resubmitBtn.click(
        re_predict, [
            user_input, 
            image_path, 
            audio_path, 
            video_path, 
            thermal_path, 
            chatbot, 
            max_length, 
            top_p, 
            temperature, 
            history, 
            modality_cache,
        ], [
            chatbot, 
            history,
            modality_cache
        ],
        show_progress=True
    )


    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[
        image_path,
        audio_path,
        video_path,
        thermal_path,
        chatbot, 
        history, 
        modality_cache
    ], show_progress=True)

demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0', server_port=24000)

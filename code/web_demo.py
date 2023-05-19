from transformers import AutoModel, AutoTokenizer
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
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.cuda().eval()
print(f'[!] init the model over ...')


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
        return [(input, "There is no image/audio/video provided. Please upload the file to start a conversation.")]
    else:
        print(f'[!] image path: {image_path}\n[!] audio path: {audio_path}\n[!] video path: {video_path}\n[!] thermal pah: {thermal_path}')
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


def reset_state():
    return None, None, None, None, [], [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">PandaGPT</h1>""")

    with gr.Row(scale=4):
        with gr.Column(scale=2):
            image_path = gr.Image(type="filepath", label="Image", value=None)

            gr.Examples(
                [ 
                    os.path.join(os.path.dirname(__file__), "./assets/images/bird_image.jpg"),
                    os.path.join(os.path.dirname(__file__), "./assets/images/dog_image.jpg"),
                    os.path.join(os.path.dirname(__file__), "./assets/images/car_image.jpg"),
                ],
                image_path
            )
        with gr.Column(scale=2):
            audio_path = gr.Audio(type="filepath", label="Audio", value=None)
            gr.Examples(
                [ 
                    os.path.join(os.path.dirname(__file__), "./assets/audios/bird_audio.wav"),
                    os.path.join(os.path.dirname(__file__), "./assets/audios/dog_audio.wav"),
                    os.path.join(os.path.dirname(__file__), "./assets/audios/car_audio.wav"),
                ],
                audio_path
            )
    with gr.Row(scale=4):
        with gr.Column(scale=2):
            video_path = gr.Video(type='file', label="Video")

            gr.Examples(
                [ 
                    os.path.join(os.path.dirname(__file__), "./assets/videos/world.mp4"),
                    os.path.join(os.path.dirname(__file__), "./assets/videos/a.mp4"),
                ],
                video_path
            )
        with gr.Column(scale=2):
            thermal_path = gr.Image(type="filepath", label="Thermal Image", value=None)

            gr.Examples(
                [ 
                    os.path.join(os.path.dirname(__file__), "./assets/thermals/190662.jpg"),
                    os.path.join(os.path.dirname(__file__), "./assets/thermals/210009.jpg"),
                ],
                thermal_path
            )

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 512, value=128, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.4, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.8, step=0.01, label="Temperature", interactive=True)

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

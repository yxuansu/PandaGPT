import torch
from collections import OrderedDict
import ipdb
import random
import json
from model.openllama import OpenLLAMAPEFTModel
import argparse
from config import *

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str, default='openllama_peft')
    parser.add_argument('--imagebind_ckpt_path', type=str)
    parser.add_argument('--delta_ckpt_path', type=str)
    parser.add_argument('--vicuna_ckpt_path', type=str)
    parser.add_argument('--max_tgt_len', type=int)
    parser.add_argument('--stage', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parser_args())
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)

    print ('Loading delta parameters...')
    delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
    print ('Delta parameter loaded.')
    model = OpenLLAMAPEFTModel(**args)
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.cuda()
    print(f'[!] Model initialized!')

    image_paths, output_texts = [], []
    # only test the first two examples
    asset_path = r'../asset/'
    random.seed(0)
    for name in ['bird', 'car', 'dog']:
    #for name in ['bird']:
        print ('-'*50)
        one_image_path = r'{}/{}_image.jpg'.format(asset_path, name)
        one_audio_path = r'{}/{}_audio.wav'.format(asset_path, name)
        print (f'[!] Image path: {one_image_path}')
        print (f'[!] Audio path: {one_audio_path}')

        prompt = 'Write a story about this to a child.'
        inputs = dict(
            image_paths=[one_image_path], 
            audio_paths=[one_audio_path],
            mode='vision',
            prompt=prompt,
            max_tgt_len=args['max_tgt_len'])
        generation, output_ids = model.generate(inputs)
        print (f'[!] Prompt: {prompt}')
        print(f'[!] Generation: {generation}')
        print (inputs['mode'])
        #print(f'[!] Mode: {inputs['mode']}')
        print(f'[!] Generation ids: {output_ids}')

        inputs['mode'] = 'audio'
        generation, output_ids = model.generate(inputs)
        print (f'[!] Prompt: {prompt}')
        print(f'[!] Generation: {generation}')
        #print(f'[!] Mode: {inputs['mode']}')
        print (inputs['mode'])
        print(f'[!] Generation ids: {output_ids}')

